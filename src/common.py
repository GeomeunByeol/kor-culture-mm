"""모든 단계가 공유하는 프롬프트, 답 정규화, 출력 규격 검사, 입출력 도구.

프롬프트 문자열을 한곳에 모아 두는 이유는 학습과 추론이 같은 문자열을 쓰도록 하기 위해서다.
"""
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------- 프롬프트

# VLM이 이미지를 직접 보고 답할 때의 시스템 프롬프트
SYSTEM = (
    "당신은 한국의 전통문화와 현대 생활문화에 정통한 전문가입니다. "
    "주어진 이미지를 자세히 관찰하고 질문에 정확하게 답하십시오."
)

# LLM이 관찰문을 읽고 답할 때의 시스템 프롬프트 (릴레이 2단계)
TXT_SYSTEM = (
    "당신은 한국의 전통문화와 현대 생활문화, 지리, 역사, 언어에 정통한 전문가입니다. "
    "이미지를 직접 볼 수는 없지만, 조사원이 이미지를 보고 기록한 관찰 내용이 주어집니다. "
    "관찰 내용과 당신의 한국문화 지식을 종합하여 질문에 정확하게 답하십시오."
)

# 문항 형식별 지시문. 선다형과 단답형은 답이 하나인지 여러 개인지에 따라 나눈다.
INSTR = {
    "MC_ONE": "위 선택지 중 정답 번호 하나만 출력하시오. 정답은 반드시 한 개이며, 여러 번호를 나열하지 마시오. 번호 외의 설명은 절대 쓰지 마시오.",
    "MC_MULTI": "위 선택지 중 정답에 해당하는 번호를 모두 골라 '/'로 구분해 오름차순으로 출력하시오(예: 1/3). 번호 외의 설명은 절대 쓰지 마시오.",
    "SA_ONE": "정답에 해당하는 단어나 구 하나만 출력하시오. 설명, 조사, 마침표를 붙이지 마시오. 후보를 여러 개 나열하지 말고 가장 확실한 답 하나만 쓰시오. '/'를 쓰지 마시오. 질문이 음절 수를 지정했다면 반드시 그 음절 수에 맞추시오.",
    "SA_MULTI": "질문이 요구하는 답들을 순서대로 '/'로 구분해 출력하시오. 설명, 조사, 마침표를 붙이지 마시오. 각 답의 음절 수가 지정됐다면 반드시 맞추시오.",
    "SA": "정답에 해당하는 단어나 구만 출력하시오. 설명, 조사, 마침표를 붙이지 마시오. 질문이 음절 수를 지정했다면 반드시 그 음절 수에 맞추시오. 정답이 여러 개면 '/'로 구분하시오.",
    "LA": "이미지에서 확인되는 정보와 한국문화 지식을 종합하여 250자 이내의 완결된 한국어 문장으로 서술하시오. 서론이나 머리말 없이 답변 내용만 쓰시오.",
}

# 문항 형식별 생성 토큰 상한
MAX_TOKENS = {"MC": 24, "SA": 32, "LA": 512}

# 참고 자료 블록의 머리말 (RAG, RAFT 공통)
REFS_HDR = "[참고 자료: 유물 사전 검색 결과. 관련 없는 항목은 무시하시오]\n"

# 단답형에서 답을 여러 개 요구하는 질문의 표지
SA_MULTI_CUE = re.compile(
    r"차례대로|차례로|각각|각\s*\d\s*음절씩|모두\s*(찾아|쓰|답|적)|순서대로|두 가지|세 가지|[23]\s*가지|[23]\s*개를|이칭\s*\d")


def is_multi(rec):
    """선다형 다중 선택 문항('모두 고르시오')인지 판정한다."""
    return bool(re.search(r"모두\s*고르", rec["model_input"]["question"]))


def sa_is_multi(rec):
    """단답형 질문이 여러 개의 답을 요구하는지 판정한다."""
    return bool(SA_MULTI_CUE.search(rec["model_input"]["question"]))


def multi_for(rec):
    """clean()에 넘길 다중 답 여부를 문항 형식에 맞게 돌려준다."""
    form = rec["metadata"]["question_form"]
    if form == "MC":
        return is_multi(rec)
    return sa_is_multi(rec) if form == "SA" else True


def instr_for(rec, sa_branch=True):
    """문항에 맞는 지시문을 고른다. sa_branch=False면 단답형에 분기 없는 지시문을 쓴다."""
    form = rec["metadata"]["question_form"]
    if form == "MC":
        return INSTR["MC_MULTI" if is_multi(rec) else "MC_ONE"]
    if form == "SA":
        if not sa_branch:
            return INSTR["SA"]
        return INSTR["SA_MULTI" if sa_is_multi(rec) else "SA_ONE"]
    return INSTR["LA"]


def question_block(rec):
    """[질문]과 [선택지] 블록을 만든다."""
    mi = rec["model_input"]
    text = "[질문]\n" + mi["question"]
    if mi.get("options"):
        text += "\n\n[선택지]\n" + "\n".join(mi["options"])
    return text


def refs_prefix(refs):
    """참고 자료가 있으면 머리말을 붙인 블록을, 없으면 빈 문자열을 돌려준다."""
    refs = (refs or "").strip()
    return REFS_HDR + refs + "\n\n" if refs else ""


def build_vlm_prompt(rec, refs=""):
    """VLM 직접 답 프롬프트: (참고 자료) + 질문 + 선택지 + 지시문."""
    mi = rec["model_input"]
    parts = [mi["question"]]
    if mi.get("options"):
        parts.append("\n".join(mi["options"]))
    parts.append(instr_for(rec))
    return refs_prefix(refs) + "\n\n".join(parts)


def build_text_prompt(rec, obs, refs="", sa_branch=True):
    """LLM 릴레이 답 프롬프트: (참고 자료) + 관찰문 + 질문 + 선택지 + 지시문."""
    obs = sanitize(obs)
    return (refs_prefix(refs) + "[이미지 관찰 기록]\n" + (obs or "(관찰 기록 없음)")
            + "\n\n" + question_block(rec) + "\n\n" + instr_for(rec, sa_branch))


# ---------------------------------------------------------------- 답 정규화

def sanitize(text):
    """VLM 출력에 섞인 <think> 태그를 없앤다.

    태그 문자열이 LLM 토크나이저에서 어휘 밖 토큰으로 바뀌어 서버가 멈춘 적이 있으므로,
    VLM에서 LLM으로 넘기는 글은 모두 이 함수를 거친다.
    """
    return re.sub(r"</?think>", "", text or "").strip()


def clean(text, form, multi=True):
    """생성 결과를 채점기가 비교하는 문자열로 정규화한다.

    선다형과 단답형은 문자열 완전 일치로 채점하므로 설명, 따옴표, 마침표가 붙으면 오답이 된다.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^<think>.*", "", text, flags=re.S).strip()
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    if form == "MC":
        nums = re.findall(r"[1-5]", text)
        if not nums:
            return text
        return "/".join(sorted(set(nums))) if multi else nums[0]
    if form == "SA":
        text = text.split("\n")[0].strip().strip('".,\'"')
        if not multi and "/" in text:
            # 답을 하나만 요구했는데 후보를 나열한 경우 첫 후보만 남긴다
            text = text.split("/")[0].strip()
        return text
    return " ".join(text.split())


# ---------------------------------------------------------------- 단답형 출력 규격

def syl(text):
    """음절 수를 센다. 한글, 숫자, 영문자만 센다."""
    return len(re.sub(r"[^가-힣0-9A-Za-z]", "", text))


def violates_spec(question, answer):
    """답이 질문의 음절·어절 규격을 어기는지 판정한다. 규격을 어긴 답은 완전 일치 채점에서 0점이다."""
    if not answer:
        return True
    syls = [int(x) for x in re.findall(r"(\d+)\s*음절", question)]
    eojs = [int(x) for x in re.findall(r"(\d+)\s*어절", question)]
    multi = bool(SA_MULTI_CUE.search(question))
    parts = answer.split("/")
    if syls:
        m = re.search(r"각\s*(\d+)\s*음절씩", question)
        if m:
            return any(syl(p) != int(m.group(1)) for p in parts)
        if len(syls) >= 2 and multi:
            return len(parts) != len(syls) or any(syl(p) != n for p, n in zip(parts, syls))
        if len(syls) == 1 and not multi:
            # "2어절 또는 3음절"처럼 대안이 주어진 질문은 어느 한쪽만 맞으면 된다
            if "또는" in question and eojs and len(answer.split()) in eojs:
                return False
            return syl(answer) != syls[0]
        return False
    if eojs:
        m = re.search(r"각\s*(\d+)\s*어절씩", question)
        if m:
            return any(len(p.split()) != int(m.group(1)) for p in parts)
        if len(eojs) >= 2 and multi:
            return len(parts) != len(eojs) or any(len(p.split()) != n for p, n in zip(parts, eojs))
        if len(eojs) == 1 and multi:
            return any(len(p.split()) != eojs[0] for p in parts)
        if len(eojs) == 1:
            return len(answer.split()) != eojs[0]
    return False


# ---------------------------------------------------------------- 입출력

def load_records(path, forms=None, limit=0):
    """대회 JSON을 읽는다. forms를 주면 해당 문항 형식만 남긴다."""
    data = json.load(open(path, encoding="utf-8"))
    if forms:
        data = [r for r in data if r["metadata"]["question_form"] in forms]
    return data[:limit] if limit else data


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_map(path, key):
    """jsonl을 {question_id: 값} 사전으로 읽는다. path가 비어 있으면 빈 사전을 돌려준다."""
    if not path:
        return {}
    return {row["question_id"]: row[key] for row in read_jsonl(path)}


def done_ids(path):
    """출력 파일에 이미 기록된 question_id 집합. 중단된 실행을 이어서 돌릴 때 쓴다."""
    return {row["question_id"] for row in read_jsonl(path)} if os.path.exists(path) else set()


def image_b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def image_message(b64, text):
    """이미지 한 장과 글로 이루어진 user 메시지 내용을 만든다."""
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": text}]


def chat(url, payload, timeout=900, retries=3):
    """서버(vLLM)에 요청한다. (응답 JSON, 오류 문자열)을 돌려준다."""
    err = ""
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json(), ""
        except Exception as e:
            err = str(e)[:200]
            time.sleep(5)
    return None, err


def chat_text(url, payload, timeout=900):
    """chat()의 첫 번째 응답 글만 돌려준다."""
    res, err = chat(url, payload, timeout)
    return ((res["choices"][0]["message"]["content"] or "") if res else ""), err


def run_parallel(todo, fn, out_path, workers=8, log_every=25):
    """문항마다 fn을 병렬로 실행하고 끝나는 대로 out_path에 덧붙인다.

    요청을 동시에 보내야 vLLM의 연속 배치가 동작한다. 완료 순서대로 기록하므로
    이어 돌리기는 위치가 아니라 question_id로 판단한다.
    """
    lock = threading.Lock()
    n = 0
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fn, rec) for rec in todo]):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if row.get("error"):
                    print(f"실패 {row['question_id']}: {row['error']}", flush=True)
                if n % log_every == 0 or n == len(todo):
                    print(f"[{n}/{len(todo)}] {row['question_id']}", flush=True)
    return n


def add_server_args(ap, url="http://127.0.0.1:8000/v1/chat/completions", model="bench"):
    """서버 주소처럼 여러 스크립트가 함께 쓰는 인자를 등록한다."""
    ap.add_argument("--url", default=url, help="chat completions 주소")
    ap.add_argument("--model", default=model, help="서버에 등록된 모델 또는 LoRA 어댑터 이름")
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}',
                    help="채팅 템플릿 인자(JSON). 추론 모드를 끄는 데 쓴다")
    ap.add_argument("--workers", type=int, default=8, help="동시 요청 수")
    ap.add_argument("--timeout", type=int, default=900, help="요청 제한 시간(초)")
