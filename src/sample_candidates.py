"""문항마다 greedy 답 하나와 확률적 생성 답 N개를 만든다.

서술형 MBR 후보(논문 3.4절)와 서술형 DPO 선호 쌍의 후보 풀(논문 3.5절)에 함께 쓴다.

  python src/sample_candidates.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms LA \
      --model a-la --n 31 --out work/la_cands_test.jsonl

출력 행: {question_id, form, model, greedy: {raw, answer, finish}, samples: [...], error}
"""
import argparse
import json

from common import (MAX_TOKENS, TXT_SYSTEM, add_server_args, build_text_prompt, chat, clean, done_ids, load_map,
                    load_records, multi_for, run_parallel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="LA")
    ap.add_argument("--rows", default="", help="build_raft_data.py의 rows 파일. 주면 거기 있는 문항만 참고 자료와 함께 생성한다")
    ap.add_argument("--n", type=int, default=31, help="확률적 생성 답의 수. greedy를 더해 후보가 N+1개가 된다")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95, help="누적 확률 상한")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="0이면 문항 형식별 기본값")
    add_server_args(ap, model="a-la")
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)
    obs = load_map(a.obs, "obs")
    refs = load_map(a.rows, "refs")

    def parse(choice, rec):
        raw = choice["message"]["content"] or ""
        form = rec["metadata"]["question_form"]
        return {"raw": raw, "answer": clean(raw, form, multi_for(rec)), "finish": choice.get("finish_reason")}

    def one(rec):
        qid, form = rec["metadata"]["question_id"], rec["metadata"]["question_form"]
        base = {"model": a.model, "max_tokens": a.max_tokens or MAX_TOKENS[form],
                "messages": [{"role": "system", "content": TXT_SYSTEM},
                             {"role": "user", "content": build_text_prompt(rec, obs.get(qid, ""), refs.get(qid, ""),
                                                                           sa_branch=bool(a.rows))}]}
        if ctk:
            base["chat_template_kwargs"] = ctk
        g, e1 = chat(a.url, {**base, "temperature": 0.0, "top_p": 1.0, "n": 1}, a.timeout)
        s, e2 = chat(a.url, {**base, "temperature": a.temperature, "top_p": a.top_p, "n": a.n, "seed": a.seed},
                     a.timeout)
        return {"question_id": qid, "form": form, "model": a.model, "error": e1 or e2,
                "greedy": parse(g["choices"][0], rec) if g else None,
                "samples": [parse(c, rec) for c in s["choices"]] if s else []}

    data = load_records(a.data, a.forms.split(","))
    if a.rows:
        data = [r for r in data if r["metadata"]["question_id"] in refs]
    done = done_ids(a.out)
    todo = [r for r in data if r["metadata"]["question_id"] not in done]
    print(f"{a.model}: 생성 대상 {len(todo)}건 (완료 {len(done)}건)", flush=True)
    run_parallel(todo, one, a.out, a.workers, log_every=10)


if __name__ == "__main__":
    main()
