"""국립국어원 공식 평가 코드(teddysum/korean_evaluation)로 예측을 채점한다.

  선다형: 정확도. 문자열 완전 일치라서 "1/2"와 "2/1"은 다른 답이다.
  단답형: 완전 일치. 정답의 '#'은 허용 답을 나누는 기호다.
  서술형: ROUGE-1(Mecab 형태소, F1)과 BLEU-1의 평균.
  종합  : 세 부문 점수의 산술 평균.

  python src/score.py --gold data/validation.json --pred work/preds_validation.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "official"))

from konlpy.tag import Mecab  # noqa: E402
from nltk.translate.bleu_score import sentence_bleu  # noqa: E402
from rouge_metric import Rouge  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

tokenizer = Mecab()


# ---- 아래 세 함수는 공식 평가 코드(evaluation.py)를 그대로 옮긴 것이다 ----
def calc_ROUGE_1(true, pred):
    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,
        weight_factor=1.0,
    )
    scores = rouge_evaluator.get_scores(pred, true)
    return scores['rouge-1']['f']


def calc_BLEU(true, pred, apply_avg=True, apply_best=False, use_mecab=True):
    stacked_bleu = []
    if type(true[0]) is str:
        true = list(map(lambda x: [x], true))
    for i in range(len(true)):
        best_bleu = 0
        sum_bleu = 0
        for j in range(len(true[i])):
            if use_mecab:
                ref = tokenizer.morphs(true[i][j])
                candi = tokenizer.morphs(pred[i])
            else:
                ref = true[i][j].split()
                candi = pred[i].split()
            score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))
            sum_bleu += score
            if score > best_bleu:
                best_bleu = score
        avg_bleu = sum_bleu / len(true[i])
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)
    return sum(stacked_bleu) / len(stacked_bleu)


def calc_exact_match(true_data, pred_data):
    correct = 0
    total = len(true_data)
    for true, pred in zip(true_data, pred_data):
        acceptable_answers = true.split('#')
        if any(pred.strip() == ans.strip() for ans in acceptable_answers):
            correct += 1
    return correct / total if total > 0 else 0
# ---- 공식 평가 코드 끝 ----


def score(gold_path, pred_path):
    gold = {r["metadata"]["question_id"]: r for r in json.load(open(gold_path, encoding="utf-8"))}
    bucket = {f: {"t": [], "p": []} for f in ("MC", "SA", "LA")}
    for line in open(pred_path, encoding="utf-8"):
        p = json.loads(line)
        g = gold[p["question_id"]]
        form = g["metadata"]["question_form"]
        bucket[form]["t"].append(g["model_output"]["answer"])
        bucket[form]["p"].append(p["answer"])

    s = {"n": {f: len(bucket[f]["t"]) for f in bucket}}
    s["accuracy"] = accuracy_score(bucket["MC"]["t"], bucket["MC"]["p"]) if bucket["MC"]["t"] else 0
    s["exact_match"] = calc_exact_match(bucket["SA"]["t"], bucket["SA"]["p"]) if bucket["SA"]["t"] else 0
    s["rouge_1"] = s["bleu"] = s["descriptive_avg"] = 0
    if bucket["LA"]["t"]:
        s["rouge_1"] = calc_ROUGE_1(bucket["LA"]["t"], bucket["LA"]["p"])
        s["bleu"] = calc_BLEU(bucket["LA"]["t"], bucket["LA"]["p"])
        s["descriptive_avg"] = (s["rouge_1"] + s["bleu"]) / 2
    s["final_score"] = (s["accuracy"] + s["exact_match"] + s["descriptive_avg"]) / 3
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="정답이 있는 문항 JSON")
    ap.add_argument("--pred", required=True, help="예측 jsonl ({question_id, answer})")
    ap.add_argument("--json-out", default="", help="점수를 저장할 JSON 경로")
    a = ap.parse_args()

    s = score(a.gold, a.pred)
    print(f"문항 수: 선다형 {s['n']['MC']} / 단답형 {s['n']['SA']} / 서술형 {s['n']['LA']}\n")
    print(f"선다형 정확도      {s['accuracy'] * 100:6.2f}")
    print(f"단답형 완전 일치   {s['exact_match'] * 100:6.2f}")
    print(f"서술형 ROUGE-1     {s['rouge_1'] * 100:6.2f}")
    print(f"서술형 BLEU-1      {s['bleu'] * 100:6.2f}")
    print(f"서술형 평균        {s['descriptive_avg'] * 100:6.2f}")
    print(f"\n종합 점수          {s['final_score'] * 100:6.2f}")
    if a.json_out:
        json.dump(s, open(a.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
