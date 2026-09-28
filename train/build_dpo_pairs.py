"""문항 형식별 DPO 선호 쌍을 만든다 (논문 3.5절, 표 3). 학습 분할만 쓴다.

단답형(sa): 선호 = 정답, 비선호 = 모델이 실제로 낸 오답.
    --wrong에 학습 분할에 대한 모델의 답 파일을 준다. 2-겹 교차 생성(한쪽 겹으로 학습한 어댑터가
    다른 겹에 답함)으로 만든 파일을 주면 표 3의 "정답/교차 오답" 구성이 된다.
서술형(la): 같은 질문에서 생성한 후보를 정답과 비교해 m = (ROUGE-1 + BLEU-1) / 2를 구한다.
    best : 선호 = 최고 후보, 비선호 = 최저 후보. 점수 차이가 --min-gap 이상인 쌍만 쓴다.
    gold : 선호 = 정답,      비선호 = 최저 후보. 최저 후보 점수가 --max-rej 이하인 쌍만 쓴다.

  python train/build_dpo_pairs.py sa --data data/train.json --obs work/obs_train.jsonl \
      --wrong work/sa_train_fold0.jsonl work/sa_train_fold1.jsonl --out work/dpo/pairs_sa.jsonl
  python train/build_dpo_pairs.py la --data data/train.json --obs work/obs_train.jsonl \
      --cands work/la_cands_train.jsonl --chosen best --out work/dpo/pairs_la.jsonl

출력 행: {question_id, question, observation, refs, chosen, rejected}
"""
import argparse
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import load_map, load_records, read_jsonl, write_jsonl  # noqa: E402

warnings.filterwarnings("ignore")


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def pairs_sa(a, data, obs):
    pairs, seen = [], set()
    for path in a.wrong:
        for row in read_jsonl(path):
            qid = row["question_id"]
            if qid not in data:
                continue
            golds = [norm(x) for x in data[qid]["model_output"]["answer"].split("#")]
            rejected = norm(row.get("answer") or row.get("raw"))
            # 빈 답, 정답과 같은 답, 이미 넣은 오답은 뺀다
            if not rejected or rejected in golds or (qid, rejected) in seen:
                continue
            seen.add((qid, rejected))
            pairs.append({"question_id": qid, "question": data[qid]["model_input"]["question"],
                          "observation": obs[qid], "refs": "", "chosen": golds[0], "rejected": rejected})
    return pairs


def pairs_la(a, data, obs):
    import score

    def m(ref, hyp):
        if not ref.strip() or not hyp.strip():
            return 0.0
        return (score.calc_ROUGE_1([ref], [hyp]) + score.calc_BLEU([ref], [hyp])) / 2 * 100

    pairs = []
    for row in read_jsonl(a.cands):
        qid = row["question_id"]
        if qid not in data:
            continue
        gold = data[qid]["model_output"]["answer"]
        cands = [(row.get("greedy") or {}).get("raw", "")] + [s.get("raw", "") for s in row.get("samples", [])]
        cands = list(dict.fromkeys(c.strip() for c in cands if c.strip()))
        if len(cands) < 2:
            continue
        scored = sorted((m(gold, c), c) for c in cands)
        (low, worst), (high, best) = scored[0], scored[-1]
        if a.chosen == "best" and high - low < a.min_gap:
            continue   # 선호가 불분명한 쌍은 뺀다
        if a.chosen == "gold" and low > a.max_rej:
            continue
        pairs.append({"question_id": qid, "question": data[qid]["model_input"]["question"],
                      "observation": obs[qid], "refs": "", "chosen": best if a.chosen == "best" else gold,
                      "rejected": worst, "m_chosen": round(high, 1), "m_rejected": round(low, 1)})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("form", choices=["sa", "la"])
    ap.add_argument("--data", required=True, help="학습 분할 문항 JSON")
    ap.add_argument("--obs", required=True, help="학습 분할의 관찰문")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrong", nargs="+", default=[], help="[sa] 학습 분할에 대한 모델의 답 파일")
    ap.add_argument("--cands", default="", help="[la] sample_candidates.py로 만든 학습 분할 후보")
    ap.add_argument("--chosen", choices=["best", "gold"], default="best", help="[la] 선호 답변의 출처")
    ap.add_argument("--min-gap", type=float, default=15.0, help="[la] 최고·최저 후보 점수 차이의 하한")
    ap.add_argument("--max-rej", type=float, default=85.0, help="[la] gold 구성에서 최저 후보 점수의 상한")
    a = ap.parse_args()

    form = a.form.upper()
    data = {r["metadata"]["question_id"]: r for r in load_records(a.data, [form])}
    obs = load_map(a.obs, "obs")
    pairs = pairs_sa(a, data, obs) if a.form == "sa" else pairs_la(a, data, obs)
    write_jsonl(a.out, pairs)
    print(f"선호 쌍 {len(pairs)}개 (문항 {len({p['question_id'] for p in pairs})}개) -> {a.out}")


if __name__ == "__main__":
    main()
