"""서술형 MBR 후보 선택 (논문 3.4절, 식 2).

생성 후보를 잠재적 참조 답변으로 보고, 다른 후보들과의 평균 효용이 가장 높은 후보를 고른다.
효용 u(a, b)는 두 후보 사이 ROUGE-1과 BLEU-1의 평균이며 양방향 값을 더해 대칭으로 만든다.
평가 정답은 쓰지 않는다. --gold를 주면 분석을 위해 greedy, MBR, 후보 중 최고 점수를 함께 출력한다.

  python src/mbr_select.py --cands work/la_cands_test.jsonl --n 32 --out work/la_mbr.jsonl
"""
import argparse
import json
import warnings
from multiprocessing import Pool

from common import read_jsonl, write_jsonl

warnings.filterwarnings("ignore")
_metric = None


def metric(ref, hyp):
    """(ROUGE-1 + BLEU-1) / 2를 100점 척도로 돌려준다. 공식 평가 코드와 같은 형태소 분석기를 쓴다."""
    global _metric
    if _metric is None:
        import score   # Mecab을 불러오므로 작업 프로세스마다 한 번만 읽는다
        _metric = score
    if not ref.strip() or not hyp.strip():
        return 0.0
    return (_metric.calc_ROUGE_1([ref], [hyp]) + _metric.calc_BLEU([ref], [hyp])) / 2 * 100


def select(job):
    qid, cands, gold = job
    n = len(cands)
    util = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            v = metric(cands[i], cands[j]) + metric(cands[j], cands[i])
            util[i] += v
            util[j] += v
    best = max(range(n), key=lambda i: util[i])
    out = {"question_id": qid, "form": "LA", "answer": cands[best], "greedy": cands[0], "n_cand": n,
           "util": round(util[best] / max(1, n - 1), 2)}
    if gold is not None:
        out.update(s_greedy=metric(gold, cands[0]), s_mbr=metric(gold, cands[best]),
                   s_oracle=max(metric(gold, c) for c in cands))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="sample_candidates.py 출력")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32, help="후보 수 N. greedy 1개와 확률적 생성 답 N-1개를 쓴다")
    ap.add_argument("--gold", default="", help="정답이 있는 문항 JSON(학습·검증). 분석 출력에만 쓴다")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    gold = {}
    if a.gold:
        gold = {r["metadata"]["question_id"]: r["model_output"]["answer"]
                for r in json.load(open(a.gold, encoding="utf-8"))}
    jobs = []
    for row in read_jsonl(a.cands):
        cands = [(row.get("greedy") or {}).get("raw", "").strip()]
        cands += [s["raw"].strip() for s in row["samples"][:max(0, a.n - 1)]]
        cands = [" ".join(c.split()) for c in cands if c]
        if cands:
            jobs.append((row["question_id"], cands, gold.get(row["question_id"])))
    with Pool(a.workers) as pool:
        rows = pool.map(select, jobs)
    write_jsonl(a.out, rows)
    if gold:
        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n
        up = sum(r["s_mbr"] > r["s_greedy"] + 1e-9 for r in rows)
        down = sum(r["s_mbr"] < r["s_greedy"] - 1e-9 for r in rows)
        print(f"문항 {n}  greedy {mean('s_greedy'):.2f}  MBR {mean('s_mbr'):.2f}  후보 중 최고 {mean('s_oracle'):.2f}  "
              f"상승 {up} 하락 {down}")
    print("->", a.out)


if __name__ == "__main__":
    main()
