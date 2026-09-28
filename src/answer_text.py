"""릴레이 2단계: LLM이 관찰문을 읽고 답한다 (논문 3.1절, y = A(q, o)).

어댑터는 vLLM 서버에 LoRA로 함께 올려 두고 --model로 바꿔 쓴다.
같은 스크립트로 세 가지 구성을 모두 실행한다.

  # SFT 어댑터 (A-SA, A-LA, A-MC')
  python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms SA \
      --model a-sa --out work/sa.jsonl
  # RAG: 게이트 문항에 참고 자료와 예제 두 개를 주고 추가 학습 없이 답한다
  python src/answer_text.py ... --model bench --rows work/raft/rows_test.jsonl \
      --shots work/raft/icl.json --out work/sa_rag.jsonl
  # RAFT: 같은 참고 자료를 검색 조건부 어댑터로 답한다
  python src/answer_text.py ... --model raft-sa --rows work/raft/rows_test.jsonl --out work/sa_raft.jsonl

출력 행: {question_id, form, raw, answer, sec, error}
"""
import argparse
import json
import time

from common import (MAX_TOKENS, TXT_SYSTEM, add_server_args, build_text_prompt, chat_text, clean, done_ids,
                    load_map, load_records, multi_for, run_parallel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True, help="observe.py 또는 cropzoom.py 출력")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="SA,LA", help="답할 문항 형식(쉼표 구분)")
    ap.add_argument("--rows", default="", help="build_raft_data.py의 rows 파일. 주면 거기 있는 문항만 참고 자료와 함께 답한다")
    ap.add_argument("--shots", default="", help="문맥 내 예제 파일(icl.json). RAG 구성에서만 쓴다")
    ap.add_argument("--sa-instr", choices=["generic", "branch"], default="generic",
                    help="단답형 지시문. generic은 분기 없는 지시문, branch는 답의 개수에 따라 나눈 지시문. "
                         "--rows를 주면 학습 때와 같도록 항상 branch를 쓴다")
    ap.add_argument("--max-tokens", type=int, default=0, help="0이면 문항 형식별 기본값")
    ap.add_argument("--limit", type=int, default=0)
    add_server_args(ap)
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)
    obs = load_map(a.obs, "obs")
    refs = load_map(a.rows, "refs")
    shots = json.load(open(a.shots, encoding="utf-8")) if a.shots else {}
    sa_branch = bool(a.rows) or a.sa_instr == "branch"

    def one(rec):
        st = time.time()
        qid, form = rec["metadata"]["question_id"], rec["metadata"]["question_form"]
        msgs = [{"role": "system", "content": TXT_SYSTEM}]
        for ex in shots.get(form, []):
            msgs += [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["gold"]}]
        msgs.append({"role": "user", "content": build_text_prompt(rec, obs.get(qid, ""), refs.get(qid, ""), sa_branch)})
        payload = {"model": a.model, "temperature": 0.0, "top_p": 1.0,
                   "max_tokens": a.max_tokens or MAX_TOKENS[form], "messages": msgs}
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
