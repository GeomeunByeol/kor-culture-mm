"""문항 형식별 최종 답을 합쳐 제출 파일을 만든다.

제출 파일은 평가 문항 JSON에 model_output.answer만 채운 것이다(들여쓰기 2, 줄바꿈 CRLF).
게이트를 통과한 문항은 검색 조건부 어댑터의 답으로, 나머지는 기존 SFT 경로의 답으로 채운다.
뒤에 적은 파일의 답이 앞의 답을 덮어쓰므로 게이트 문항의 답 파일을 뒤에 둔다.

  python src/make_submission.py --data data/test.json \
      --preds work/mc_final.jsonl work/sa_final.jsonl work/la_mbr.jsonl work/sa_raft_final.jsonl work/la_raft_mbr.jsonl \
      --out submission.json
"""
import argparse
import json
import re
import sys
from collections import OrderedDict

from common import is_multi, read_jsonl


def enforce(form, ans, multi, a):
    """제출 형식을 지키도록 답을 다듬는다."""
    ans = (ans or "").strip()
    if form == "MC":
        nums = re.findall(r"[1-5]", ans)
        if not nums:
            return a.mc_fallback
        return "/".join(sorted(set(nums))) if multi else nums[0]
    if form == "SA":
        s = ans.split("\n")[0].strip().strip('"\'“”‘’「」『』.。')
        return s[:a.sa_max_chars] or a.empty_fallback
    s = " ".join(ans.split())
    if len(s) > a.la_max_chars:
        # 글자 수 상한 안의 마지막 문장 경계에서 자른다
        cut = s[:a.la_max_chars]
        m = max(cut.rfind("다."), cut.rfind("요."), cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        s = cut[:m + 1] if m >= a.la_max_chars // 2 else cut.rstrip()
    return s or a.empty_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="평가 문항 JSON")
    ap.add_argument("--preds", nargs="+", required=True, help="답 jsonl 목록. 뒤의 파일이 앞의 답을 덮어쓴다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--la-max-chars", type=int, default=250, help="서술형 글자 수 상한")
    ap.add_argument("--sa-max-chars", type=int, default=40, help="단답형 글자 수 상한")
    ap.add_argument("--mc-fallback", default="1", help="선다형 답에서 번호를 찾지 못했을 때 쓸 답")
    ap.add_argument("--empty-fallback", default="모름", help="단답형·서술형 답이 비었을 때 쓸 답")
    a = ap.parse_args()

    preds = {}
    for path in a.preds:
        for row in read_jsonl(path):
            preds[row["question_id"]] = row["answer"]
    data = json.load(open(a.data, encoding="utf-8"), object_pairs_hook=OrderedDict)
    missing = [r["metadata"]["question_id"] for r in data if r["metadata"]["question_id"] not in preds]
    if missing:
        sys.exit(f"답이 없는 문항 {len(missing)}건 (예: {missing[:5]})")
    for r in data:
        form = r["metadata"]["question_form"]
        r["model_output"] = OrderedDict(
            [("answer", enforce(form, preds[r["metadata"]["question_id"]], is_multi(r), a))])
    text = json.dumps(data, ensure_ascii=False, indent=2).replace("\n", "\r\n")
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"{a.out}: {len(data)}문항")


if __name__ == "__main__":
    main()
