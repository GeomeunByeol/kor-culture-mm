"""문항 형식별 SFT 학습 데이터를 학습 분할에서 만든다 (논문 3.2절).

행의 종류는 세 가지다.
  format  : 추론과 똑같은 프롬프트 -> 정답 (문제 풀이)
  prose   : 질문 -> 정답을 담은 서술문 (정답 선택지의 문장 변환)
  caption : 이미지 -> 정답에 근거한 설명문 (VLM 전용)

어댑터별 구성은 다음과 같다.
  vlm_mc.jsonl   A-MC  : 전 문항의 format + prose + caption (이미지 입력, 1,000문항 x 3 = 3,000행)
  text_mc.jsonl  A-MC' : 선다형 format + 전 문항 prose (관찰문 입력)
  text_sa.jsonl  A-SA  : 단답형 format (관찰문 + 질문 -> 정답)
  text_la.jsonl  A-LA  : 서술형 질문 -> 정답. 관찰문은 넣지 않고 추론 때만 준다.

  python train/build_sft_data.py --data data/train.json --obs work/obs_train.jsonl --out-dir work/sft
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import (INSTR, SYSTEM, TXT_SYSTEM, build_text_prompt, build_vlm_prompt, load_map, load_records,  # noqa: E402
                    question_block, write_jsonl)

PROSE_SYSTEM = ("당신은 한국의 전통문화와 현대 생활문화, 지리, 역사, 언어에 정통한 전문가입니다. "
                "질문에 한 문장으로 정확하게 답하십시오.")
PROSE_INSTR = "질문에 대한 답을 완결된 서술문으로 쓰시오."
CAPTION_SYSTEM = "당신은 이미지를 정밀하게 관찰하여 기록하는 조사원입니다. 보이는 사실을 한국어로 빠짐없이 적으십시오."
CAPTION_INSTR = "이 이미지를 한국 문화의 관점에서 설명하시오. 글씨가 있으면 원문 그대로 적고, 사물의 이름과 용도를 밝히시오."


def josa(word, with_final, without_final):
    """받침 유무에 맞는 조사를 고른다."""
    if not word or not ("가" <= word[-1] <= "힣"):
        return without_final
    return with_final if (ord(word[-1]) - 0xAC00) % 28 else without_final


def option_text(rec, num):
    for opt in rec["model_input"]["options"]:
        m = re.match(r"\s*(\d+)\s*[)\.]\s*(.*)", opt)
        if m and m.group(1) == str(num):
            return m.group(2).strip()
    return ""


def mc_prose(rec):
    """정답 선택지를 서술문으로 바꾼다. '옳지 않은 것'을 묻는 문항은 부정문으로 쓴다."""
    negative = bool(re.search(r"옳지\s*않|틀린|아닌\s*것|맞지\s*않", rec["model_input"]["question"]))
    sents = []
    for num in rec["model_output"]["answer"].split("/"):
        text = option_text(rec, num.strip()).rstrip(".。")
        if not text:
            continue
        if negative:
            sents.append(f"'{text}'는 사실이 아니다.")
        else:
            sents.append(text + ("" if text.endswith("다") else "이다") + ".")
    return " ".join(sents)


def sa_prose(rec):
    """단답형 정답을 '<질문 대상>은/는 <정답>이다.' 꼴의 서술문으로 바꾼다."""
    gold = rec["model_output"]["answer"].split("#")[0]
    q = rec["model_input"]["question"]
    # 답 형식을 지정하는 꼬리("...을 2음절로 답하시오")를 떼어 질문 대상만 남긴다
    subj = re.sub(r"(을|를|은|는|이|가)?\s*(\d+\s*(음절|어절)\s*(로|으로)\s*)?(답하시오|쓰시오|적으시오|답하십시오|고르시오).*$", "", q).strip()
    subj = re.sub(r"(무엇인지|무엇이라\s*하는지|무엇을\s*의미하는지|어디인지|누구인지|몇\s*\S+인지)\s*$", "", subj).strip(" ,?")
    subj = re.sub(r"\s*(이|가|은|는|을|를)$", "", subj).strip()
    if "/" in gold:
        return f"{subj}: 차례대로 {gold.replace('/', ', ')}이다."
    return f"{subj}{josa(subj, '은', '는')} {gold}이다." if subj else f"정답은 {gold}이다."


def row(kind, image, system, user, answer):
    return {"kind": kind, "image": image,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="학습 분할 문항 JSON")
    ap.add_argument("--obs", required=True, help="학습 분할의 관찰문(observe.py 출력)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    obs = load_map(a.obs, "obs")

    out = {"vlm_mc": [], "text_mc": [], "text_sa": [], "text_la": []}
    for rec in load_records(a.data):
        qid, form = rec["metadata"]["question_id"], rec["metadata"]["question_form"]
        gold, img = rec["model_output"]["answer"], rec["model_input"]["image_name"]
        o = obs.get(qid, "").strip()
        prose = {"MC": mc_prose, "SA": sa_prose}[form](rec) if form != "LA" else gold

        # format: 추론과 같은 프롬프트
        out["vlm_mc"].append(row(f"format-{form}", img, SYSTEM, build_vlm_prompt(rec), gold))
        if form == "MC":
            out["text_mc"].append(row("format-MC", None, TXT_SYSTEM, build_text_prompt(rec, o), gold))
        elif form == "SA":
            out["text_sa"].append(row("format-SA", None, TXT_SYSTEM, build_text_prompt(rec, o), gold))
        else:
            out["text_la"].append(row("format-LA", None, TXT_SYSTEM,
                                      "[질문]\n" + rec["model_input"]["question"] + "\n\n" + INSTR["LA"], gold))

        # prose: 정답을 담은 서술문
        if prose:
            user = question_block(rec) + "\n\n" + PROSE_INSTR
            out["vlm_mc"].append(row(f"prose-{form}", img, PROSE_SYSTEM, user, prose))
            out["text_mc"].append(row(f"prose-{form}", None, PROSE_SYSTEM,
                                      ("[이미지 관찰 기록]\n" + o + "\n\n" if o else "") + user, prose))

        # caption: 서술형은 정답 서술문 그대로, 나머지는 정답 서술문에 관찰문을 잇는다
        cap = gold if form == "LA" else (prose + " " if prose else "") + o
        if cap.strip():
            out["vlm_mc"].append(row(f"caption-{form}", img, CAPTION_SYSTEM, CAPTION_INSTR, cap.strip()))

    for name, rows in out.items():
        random.shuffle(rows)
        write_jsonl(os.path.join(a.out_dir, name + ".jsonl"), rows)
        print(name, len(rows), dict(Counter(r["kind"] for r in rows)))


if __name__ == "__main__":
    main()
