"""검색 조건부 미세조정(RAFT) 학습·추론 데이터를 만든다 (논문 3.3절).

게이트를 통과한 문항의 입력 앞에 [참고 자료] 블록을 붙인다.
원 논문의 방해 문서 혼합과 근거 인용형 추론 사슬은 쓰지 않고, 참고 자료와 함께 정답을 생성하도록만 학습한다.
추론 때에도 게이트 문항에 같은 입력 구성을 쓴다.

  python train/build_raft_data.py --split train --data data/train.json --obs work/obs_train.jsonl \
      --gate work/gate_train.json --out-dir work/raft
  python train/build_raft_data.py --split test --data data/test.json --obs work/obs_test_zoom.jsonl \
      --gate work/gate_test.json --out-dir work/raft

출력:
  rows_{split}.jsonl                  추론용 {question_id, form, image, refs, gold}
  (train) text_train_{mc,sa,la}.jsonl LLM 학습 데이터 -> train_lora_text.py --init-adapter <SFT 어댑터>
  (train) vlm_train_mc.jsonl          VLM 학습 데이터 -> train_lora_vlm.py --init-adapter <A-MC>
  (train) icl.json, icl_vlm.json      RAG 비교용 문맥 내 예제. 문항 형식마다 --n-shots개
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import (SYSTEM, TXT_SYSTEM, build_text_prompt, build_vlm_prompt, load_map, load_records,  # noqa: E402
                    write_jsonl)


def messages(system, user, gold):
    return [{"role": "system", "content": system}, {"role": "user", "content": user},
            {"role": "assistant", "content": gold}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--gate", required=True, help="build_gate.py 출력")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-shots", type=int, default=2, help="RAG 구성에 줄 문맥 내 예제 수")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    gate = json.load(open(a.gate, encoding="utf-8"))
    obs = load_map(a.obs, "obs")

    rows = []
    train = {"text_train_mc": [], "text_train_sa": [], "text_train_la": [], "vlm_train_mc": []}
    icl, icl_vlm = {"MC": [], "SA": [], "LA": []}, {"MC": []}
    for rec in load_records(a.data):
        qid, form = rec["metadata"]["question_id"], rec["metadata"]["question_form"]
        if qid not in gate:
            continue
        refs = gate[qid]["refs"]
        gold = rec.get("model_output", {}).get("answer", "")
        image = rec["model_input"]["image_name"]
        rows.append({"question_id": qid, "form": form, "image": image, "refs": refs, "gold": gold})
        if a.split != "train":   # 학습 데이터와 예제는 학습 분할에서만 만든다
            continue
        user_text = build_text_prompt(rec, obs.get(qid, ""), refs)
        user_vlm = build_vlm_prompt(rec, refs)
        train[f"text_train_{form.lower()}"].append(
            {"kind": f"raft-{form}", "image": None, "messages": messages(TXT_SYSTEM, user_text, gold)})
        if form == "MC":
            train["vlm_train_mc"].append(
                {"kind": "raft-MC", "image": image, "messages": messages(SYSTEM, user_vlm, gold)})
        if refs and len(icl[form]) < a.n_shots:
            icl[form].append({"user": user_text, "gold": gold})
            if form == "MC":
                icl_vlm["MC"].append({"user": user_vlm, "gold": gold})

    write_jsonl(os.path.join(a.out_dir, f"rows_{a.split}.jsonl"), rows)
    print(f"rows_{a.split}: {len(rows)}")
    if a.split == "train":
        for name, data in train.items():
            write_jsonl(os.path.join(a.out_dir, name + ".jsonl"), data)
            print(name, len(data))
        for name, shots in (("icl.json", icl), ("icl_vlm.json", icl_vlm)):
            json.dump(shots, open(os.path.join(a.out_dir, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
