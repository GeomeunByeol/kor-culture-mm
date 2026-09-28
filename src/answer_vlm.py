"""VLM이 이미지를 직접 보고 답한다. 선다형 투표의 구성원(원본 VLM, 선다형 어댑터 A-MC)에 쓴다.

  # 원본 VLM
  python src/answer_vlm.py --data data/test.json --images data/images/test_1280 --forms MC \
      --model bench --out work/mc_base.jsonl
  # 선다형 어댑터(A-MC)
  python src/answer_vlm.py ... --model a-mc --out work/mc_sft.jsonl
  # 게이트 문항: 참고 자료를 붙여 검색 조건부 어댑터로 답한다
  python src/answer_vlm.py ... --model raft-mc --rows work/raft/rows_test.jsonl --out work/mc_raft.jsonl

출력 행: {question_id, form, raw, answer, sec, error}
"""
import argparse
import json
import os
import time

from common import (MAX_TOKENS, SYSTEM, add_server_args, build_vlm_prompt, chat_text, clean, done_ids,
                    image_b64, image_message, load_map, load_records, multi_for, run_parallel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True, help="EXIF 보정된 1,280픽셀 이미지 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="MC", help="답할 문항 형식(쉼표 구분)")
    ap.add_argument("--rows", default="", help="build_raft_data.py의 rows 파일. 주면 거기 있는 문항만 참고 자료와 함께 답한다")
    ap.add_argument("--shots", default="", help="문맥 내 예제 파일(icl_vlm.json). RAG 구성에서만 쓴다")
    ap.add_argument("--limit", type=int, default=0)
    add_server_args(ap)
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)
    refs = load_map(a.rows, "refs")
    shots = json.load(open(a.shots, encoding="utf-8")) if a.shots else {}

    def one(rec):
        st = time.time()
        qid, form = rec["metadata"]["question_id"], rec["metadata"]["question_form"]
        img = os.path.join(a.images, rec["model_input"]["image_name"])
        msgs = [{"role": "system", "content": SYSTEM}]
        for ex in shots.get(form, []):   # 예제는 이미지 없이 글로만 준다
            msgs += [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["gold"]}]
        msgs.append({"role": "user", "content": image_message(image_b64(img), build_vlm_prompt(rec, refs.get(qid, "")))})
        payload = {"model": a.model, "temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_TOKENS[form],
                   "messages": msgs}
        if ctk:
            payload["chat_template_kwargs"] = ctk
        raw, err = chat_text(a.url, payload, a.timeout)
        return {"question_id": qid, "form": form, "raw": raw, "answer": clean(raw, form, multi_for(rec)),
                "sec": round(time.time() - st, 2), "error": err}

    data = load_records(a.data, a.forms.split(","), a.limit)
    if a.rows:
        data = [r for r in data if r["metadata"]["question_id"] in refs]
    done = done_ids(a.out)
    todo = [r for r in data if r["metadata"]["question_id"] not in done]
    print(f"답할 문항 {len(todo)}건 (완료 {len(done)}건)", flush=True)
    run_parallel(todo, one, a.out, a.workers)


if __name__ == "__main__":
    main()
