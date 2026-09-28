"""단답형 출력 규격 교정 (논문 4.2절: 음절·어절 제약에 따른 교정, 재시도, 후보 필터).

규격을 어긴 답은 완전 일치 채점에서 반드시 0점이므로, 규격을 어긴 답만 고친다.
규격을 지킨 답은 건드리지 않는다. 맞은 답을 동의어로 바꿔 쓰는 손실을 막기 위해서다.

  1) 음절 재시도: 자기 답과 음절 수를 보여 주고 다시 묻는다. 실패하면 규격에 맞는 후보를 나열하게 한다.
  2) 후보 필터: 여전히 규격을 어기면 다른 구성의 답 가운데 규격을 지키는 첫 후보로 바꾼다.
  3) 어절 재시도: 어절 수를 어긴 답을 다시 묻는다.

  python src/sa_postprocess.py --data data/test.json --obs work/obs_test_zoom.jsonl --preds work/sa.jsonl \
      --cands work/sa_other.jsonl --model a-sa --out work/sa_final.jsonl
"""
import argparse
import json
import re

from common import (SA_MULTI_CUE, TXT_SYSTEM, add_server_args, build_text_prompt, chat_text, clean, load_map,
                    load_records, multi_for, read_jsonl, syl, violates_spec, write_jsonl)

# "정답: X" 꼴의 머리말. 음절 수를 셀 때 섞이지 않도록 먼저 없앤다.
ANSWER_TAG = re.compile(r"^\**\s*정\s*답\s*\**\s*[:：]\s*")


def untag(text):
    return ANSWER_TAG.sub("", (text or "").strip()).strip()


def single_spec(question, unit):
    """질문이 답 하나에 대해 unit(음절 또는 어절) 수를 한 번만 지정했으면 그 수를 돌려준다."""
    nums = re.findall(r"(\d+)\s*" + unit, question)
    return int(nums[0]) if len(nums) == 1 and not SA_MULTI_CUE.search(question) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--preds", required=True, help="answer_text.py의 단답형 출력")
    ap.add_argument("--cands", default="", help="후보 필터에 쓸 다른 답 파일(쉼표 구분, 앞쪽이 우선)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--list-max-tokens", type=int, default=96, help="후보 나열 재시도의 토큰 상한")
    ap.add_argument("--n-list", type=int, default=3, help="후보 나열 재시도에서 요구할 후보 수")
    add_server_args(ap, model="a-sa")
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)
    recs = {r["metadata"]["question_id"]: r for r in load_records(a.data, ["SA"])}
    obs = load_map(a.obs, "obs")
    pools = [load_map(p, "answer") for p in a.cands.split(",") if p]

    def ask(rec, followup, max_tokens):
        payload = {"model": a.model, "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": TXT_SYSTEM},
                                {"role": "user", "content": build_text_prompt(
                                    rec, obs.get(rec["metadata"]["question_id"], ""), sa_branch=False) + followup}]}
        if ctk:
            payload["chat_template_kwargs"] = ctk
        return chat_text(a.url, payload, a.timeout)[0]

    def retry_syllable(rec, ans, n):
        """음절 재시도. 규격에 맞는 새 답을 돌려주고, 끝내 맞추지 못하면 빈 문자열을 돌려준다."""
        raw = ask(rec, f"\n\n[검토] 앞서 '{ans}'라고 답했으나 이는 {syl(ans)}음절입니다. 질문은 정확히 {n}음절을 요구합니다. "
                       f"지역명·수식어·조사·띄어쓰기를 빼고 핵심 명칭만 남기거나, 더 정확한 명칭을 떠올려 정확히 {n}음절로 다시 답하시오. "
                       f"답만 출력하시오.", a.max_tokens)
        new = clean(raw, "SA", False).replace(" ", "")
        if new and syl(new) == n:
            return new
        # 규격을 검색 조건으로 삼아 후보를 나열하게 한 뒤 규격에 맞는 첫 후보를 고른다
        raw = ask(rec, f"\n\n[검토] 질문의 정답은 정확히 {n}음절이어야 합니다. 질문이 요구하는 종류(예: 도형 이름, 기관명, 지명, 물건 이름)에 "
                       f"맞으면서 정확히 {n}음절인 서로 다른 후보를 {a.n_list}개 나열하시오. 각 후보는 한 줄에 하나씩 쓰시오. "
                       f"마지막 줄에 가장 옳은 하나를 '정답: 후보' 형식으로 쓰시오.", a.list_max_tokens)
        m = re.search(r"정답\s*[:：]\s*(.+)", raw)
        cands = ([m.group(1)] if m else []) + [l.strip(" -*0123456789.)") for l in raw.splitlines() if l.strip()]
        return next((c.replace(" ", "").strip("'\"") for c in cands if c and syl(c.replace(" ", "")) == n), "")

    def retry_eojeol(rec, ans, n):
        """어절 재시도. 규격에 맞는 새 답을 돌려주고, 맞추지 못하면 빈 문자열을 돌려준다."""
        raw = ask(rec, f"\n\n[검토] 앞서 '{ans}'라고 답했으나 이는 {len(ans.split())}어절입니다. "
                       f"질문은 정확히 {n}어절(띄어쓰기 기준 {n}개 단어)을 요구합니다. "
                       f"같은 명칭이면 띄어쓰기만 조정해 정확히 {n}어절로 다시 쓰시오. "
                       f"띄어쓰기 조정만으로 안 되면 명칭을 다시 생각해 정확히 {n}어절로 답하시오. 답만 출력하시오.", a.max_tokens)
        new = clean(raw, "SA", multi_for(rec))
        return new if len(new.split()) == n else ""

    rows = read_jsonl(a.preds)
    stat = {"syllable": 0, "filter": 0, "eojeol": 0, "residual": 0}
    for row in rows:
        if row.get("form", "SA") != "SA":
            continue
        rec = recs[row["question_id"]]
        q = rec["model_input"]["question"]
        ans = untag(row["answer"])

        n_syl = single_spec(q, "음절")
        if n_syl and syl(ans) == n_syl:
            ans = ans.replace(" ", "")   # 음절 수를 지정한 문항의 정답은 붙여 쓴다
        elif n_syl and violates_spec(q, ans):
            new = retry_syllable(rec, ans, n_syl)
            if new:
                ans, stat["syllable"] = new, stat["syllable"] + 1

        if violates_spec(q, ans):
            cands = [untag(p[row["question_id"]]) for p in pools if p.get(row["question_id"])]
            key = re.sub(r"\s+", "", ans)
            # 글자는 같고 띄어쓰기만 다른 후보를 먼저 찾고, 없으면 규격을 지키는 첫 후보를 쓴다
            pick = (next((c for c in cands if re.sub(r"\s+", "", c) == key and not violates_spec(q, c)), None)
                    or next((c for c in cands if not violates_spec(q, c)), None))
            if pick:
                ans, stat["filter"] = pick, stat["filter"] + 1

        n_eoj = single_spec(q, "어절")
        if n_eoj and violates_spec(q, ans):
            new = retry_eojeol(rec, ans, n_eoj)
            if new:
                ans, stat["eojeol"] = new, stat["eojeol"] + 1

        if ans != row["answer"]:
            row["answer_before_postprocess"], row["answer"] = row["answer"], ans
        stat["residual"] += violates_spec(q, ans)
    write_jsonl(a.out, rows)
    print(stat, "->", a.out, flush=True)


if __name__ == "__main__":
    main()
