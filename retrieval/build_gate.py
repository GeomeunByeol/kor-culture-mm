"""선택적 검색 게이트와 참고 자료 구성 (논문 3.3절, 4.2절).

게이트: 질문·선택지·관찰문에 유물 관련 용어가 있거나, 이미지 검색 최고 유사도가 임곗값보다 크면 통과한다.
참고 자료: 게이트를 통과한 문항에만 만든다.
  1) 유사도가 임곗값을 넘는 상위 이미지의 표제어와 설명
  2) 상위 이미지의 표제어와 질문으로 사전·백과사전을 검색한 상위 문서 조각(BM25 + BGE-M3, RRF)

  python retrieval/build_gate.py --data data/test.json --obs work/obs_test_zoom.jsonl \
      --image-hits work/img_hits_test.jsonl --index-dir work/txt_index \
      --keywords configs/gate_keywords.example.json --out work/gate_test.json

출력: {question_id: {form, sim, by_keyword, by_image, refs}}  (게이트를 통과한 문항만)
--sim-threshold를 바꾸고 --no-keyword를 주면 유사도 임곗값 실험(논문 5.3절)의 적용 범위를 만들 수 있다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import load_map, load_records  # noqa: E402
from text_index import Retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--image-hits", required=True, help="image_index.py query 출력")
    ap.add_argument("--index-dir", required=True, help="text_index.py build로 만든 글 색인")
    ap.add_argument("--keywords", default="", help="유물 관련 용어 목록(JSON 배열)")
    ap.add_argument("--no-keyword", action="store_true", help="용어 조건을 끄고 이미지 유사도만으로 게이트를 연다")
    ap.add_argument("--sim-threshold", type=float, default=0.88, help="이미지 최고 유사도 임곗값")
    ap.add_argument("--top-chunks", type=int, default=2, help="표제어마다 고를 문서 조각 수")
    ap.add_argument("--obs-chars", type=int, default=400, help="용어 조건과 검색 질의에 쓸 관찰문 앞부분 길이")
    ap.add_argument("--chunk-cap", type=int, default=3500, help="조각 하나의 글자 수 상한")
    ap.add_argument("--total-cap", type=int, default=5500, help="참고 자료 전체의 글자 수 상한")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    terms = [] if (a.no_keyword or not a.keywords) else json.load(open(a.keywords, encoding="utf-8"))
    obs = load_map(a.obs, "obs")
    hits = load_map(a.image_hits, "hits")
    retriever = Retriever(a.index_dir, rrf_k=a.rrf_k, pool=a.pool)

    gate = {}
    for rec in load_records(a.data):
        qid, mi = rec["metadata"]["question_id"], rec["model_input"]
        text = mi["question"] + " " + " ".join(mi.get("options") or []) + " " + obs.get(qid, "")[:a.obs_chars]
        top = [h for h in hits.get(qid, []) if h["score"] > a.sim_threshold]
        sim = max((h["score"] for h in hits.get(qid, [])), default=0.0)
        by_keyword = any(t in text for t in terms)
        if not (by_keyword or top):
            continue

        lines = [f"- ({h['source']} 이미지 검색) {h['name']}: {h['desc']}" for h in top[:1]]
        headwords = [h["name"] for h in top[:1] if h["name"]]
        query = mi["question"] + " " + " ".join(mi.get("options") or [])
        # 표제어가 있으면 그 항목 안에서, 없으면 색인 전체에서 질문과 가까운 조각을 고른다
        for hw in headwords or [None]:
            q = query if hw else query + "\n" + obs.get(qid, "")[:a.obs_chars]
            for c in retriever.search(q, a.top_chunks, [hw] if hw else None):
                lines.append(f"- ({c['source']}) {c['text'][:a.chunk_cap]}")
        refs, used = [], 0
        for line in lines:
            if used + len(line) > a.total_cap:
                break
            refs.append(line)
            used += len(line)
        gate[qid] = {"form": rec["metadata"]["question_form"], "sim": round(sim, 4),
                     "by_keyword": by_keyword, "by_image": bool(top), "refs": "\n".join(refs)}

    json.dump(gate, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_form = {f: sum(v["form"] == f for v in gate.values()) for f in ("MC", "SA", "LA")}
    print(f"게이트 통과 {len(gate)}문항 {n_form} -> {a.out}")


if __name__ == "__main__":
    main()
