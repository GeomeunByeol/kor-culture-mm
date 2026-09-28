"""선다형 답 결합 (논문 4.2절).

단일 선택 문항: 선다형 어댑터 VLM, 원본 VLM, 릴레이 답의 다수결. 동률이면 선다형 어댑터의 답을 따른다.
다중 선택 문항: 텍스트 어댑터(A-MC')의 답을 쓰되, 그 답이 두 개짜리 집합이고
              투표 상위 두 선택지와 다를 때만 투표 결과로 바꾼다(재투표).
게이트 문항: 검색 조건부 어댑터의 답으로 바꾼다.

  python src/mc_vote.py --data data/test.json --sft work/mc_sft.jsonl --base work/mc_base.jsonl \
      --relay work/mc_relay.jsonl --multi work/mc_text.jsonl --raft work/mc_raft.jsonl --out work/mc_final.jsonl
"""
import argparse
from collections import Counter

from common import is_multi, load_map, load_records, write_jsonl


def majority(answers):
    """답 문자열의 다수결. answers[0]이 선다형 어댑터의 답이며 동률이면 이것을 고른다."""
    count = Counter(x for x in answers if x)
    if not count:
        return ""
    best = max(count.values())
    winners = [x for x in count if count[x] == best]
    return answers[0] if answers[0] in winners else winners[0]


def top_options(answers, k):
    """선택지 단위 득표 상위 k개. 득표가 같으면 선다형 어댑터가 고른 선택지, 번호가 작은 선택지 순이다."""
    votes = Counter(n for x in answers for n in set(x.split("/")) if n.isdigit())
    first = set(answers[0].split("/"))
    order = sorted(votes, key=lambda n: (-votes[n], n not in first, int(n)))
    return sorted(order[:k], key=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sft", required=True, help="선다형 어댑터(A-MC) VLM의 답")
    ap.add_argument("--base", required=True, help="원본 VLM의 답")
    ap.add_argument("--relay", required=True, help="관찰문을 읽은 LLM의 답")
    ap.add_argument("--multi", default="", help="다중 선택 문항용 텍스트 어댑터(A-MC')의 답")
    ap.add_argument("--raft", default="", help="게이트 문항용 검색 조건부 어댑터의 답")
    ap.add_argument("--revote-size", type=int, default=2, help="재투표를 적용할 답 집합의 크기")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    members = [load_map(p, "answer") for p in (a.sft, a.base, a.relay)]
    multi, raft = load_map(a.multi, "answer"), load_map(a.raft, "answer")
    rows, stat = [], Counter()
    for rec in load_records(a.data, ["MC"]):
        qid = rec["metadata"]["question_id"]
        answers = [m.get(qid, "") for m in members]
        ans, src = majority(answers), "vote"
        if qid in raft:
            ans, src = raft[qid], "raft"
        if is_multi(rec) and multi.get(qid):
            ans, src = multi[qid], "multi"
            picked = ans.split("/")
            top = top_options(answers, a.revote_size)
            if len(picked) == a.revote_size and len(top) == a.revote_size and picked != top:
                ans, src = "/".join(top), "revote"
        stat[src] += 1
        rows.append({"question_id": qid, "form": "MC", "answer": ans, "source": src, "members": answers})
    write_jsonl(a.out, rows)
    print(dict(stat), "->", a.out)


if __name__ == "__main__":
    main()
