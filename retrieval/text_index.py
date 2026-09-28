"""사전·백과사전 글 색인과 혼합 검색기 (논문 4.2절).

BM25(Mecab 형태소)와 BGE-M3 밀집 검색의 순위를 Reciprocal Rank Fusion으로 합친다.
외부 API를 쓰지 않고 모두 로컬에서 실행한다.

  python retrieval/text_index.py build --corpus ext/articles.jsonl --index-dir work/txt_index
  python retrieval/text_index.py search --index-dir work/txt_index --query "곡식을 까불러 쭉정이를 걸러내는 도구"

corpus 행: {"id", "headword", "definition", "body", "source"}  (source 예: 우리말샘, 민백)
조각(chunk)은 두 종류다.
  head : "표제어 정의" 한 줄. 표제어 자체가 답인 짧은 질문에 맞는다.
  body : 본문을 문단 단위로 묶은 조각. 앞에 표제어를 붙인다.
"""
import argparse
import json
import os
import re
import sys

import faiss
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "official"))
from konlpy.tag import Mecab  # noqa: E402


def make_chunks(corpus_path, chunk_chars, min_chars):
    chunks = []
    for line in open(corpus_path, encoding="utf-8"):
        d = json.loads(line)
        hw = (d.get("headword") or "").strip()
        if not hw:
            continue
        base = {"id": d["id"], "hw": hw, "source": d.get("source", "")}
        chunks.append({**base, "kind": "head", "text": f"{hw} {(d.get('definition') or '').strip()}".strip()})
        cur = ""
        paras = [p.strip() for p in re.split(r"\n+", d.get("body") or "") if p.strip()]
        for p in paras + [None]:
            # 문단을 이어 붙이다가 목표 길이를 넘으면 조각을 끊는다
            if p is None or (cur and len(cur) + len(p) > chunk_chars):
                if len(cur) >= min_chars:
                    chunks.append({**base, "kind": "body", "text": f"{hw}: {cur}"})
                cur = ""
            if p:
                cur = (cur + "\n" + p).strip()
    return chunks


def build_bm25(chunks, tok, k1, b):
    """BM25 가중치를 (조각 x 어휘) 희소 행렬로 미리 계산한다. 질의 점수는 해당 열의 합으로 바로 구한다."""
    vocab, rows, cols, tfs = {}, [], [], []
    dl = np.zeros(len(chunks), dtype=np.float32)
    for i, c in enumerate(chunks):
        terms = tok.morphs(c["text"])
        dl[i] = len(terms)
        counts = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
        for t, tf in counts.items():
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
            tfs.append(tf)
    tf = sparse.coo_matrix((np.array(tfs, dtype=np.float32), (rows, cols)), shape=(len(chunks), len(vocab)))
    df = np.bincount(tf.col, minlength=len(vocab))
    idf = np.log(1 + (len(chunks) - df + 0.5) / (df + 0.5)).astype(np.float32)
    denom = tf.data + k1 * (1 - b + b * dl[tf.row] / float(dl.mean()))
    weight = idf[tf.col] * tf.data * (k1 + 1) / denom
    return sparse.csr_matrix((weight.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).tocsc(), vocab


def encode(model, texts, batch, max_length):
    vecs = model.encode(texts, batch_size=batch, max_length=max_length, return_dense=True,
                        return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
    vecs = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


class Retriever:
    def __init__(self, index_dir, embed_model="BAAI/bge-m3", rrf_k=60, pool=50, max_length=512):
        from FlagEmbedding import BGEM3FlagModel
        self.chunks = [json.loads(line) for line in open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8")]
        self.bm25 = sparse.load_npz(os.path.join(index_dir, "bm25.npz"))
        self.vocab = json.load(open(os.path.join(index_dir, "bm25_vocab.json"), encoding="utf-8"))
        self.dense = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
        self.model = BGEM3FlagModel(embed_model, use_fp16=True)
        self.tok = Mecab()
        self.rrf_k, self.pool, self.max_length = rrf_k, pool, max_length

    def bm25_top(self, query):
        cols = [self.vocab[t] for t in set(self.tok.morphs(query)) if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.bm25[:, cols].sum(axis=1)).ravel()
        return [int(i) for i in np.argsort(scores)[::-1][:self.pool] if scores[i] > 0]

    def dense_top(self, query):
        _, ids = self.dense.search(encode(self.model, [query], 1, self.max_length), self.pool)
        return [int(i) for i in ids[0] if i >= 0]

    def search(self, query, k=2, headwords=None):
        """두 검색기의 순위를 RRF로 합쳐 상위 k개 조각을 돌려준다.

        headwords를 주면 표제어가 그 안에 있는 조각만 남긴다.
        """
        allow = {normalize(h) for h in headwords} if headwords else None
        rrf = {}
        for ranking in (self.bm25_top(query), self.dense_top(query)):
            rank = 0
            for idx in ranking:
                if allow is not None and normalize(self.chunks[idx]["hw"]) not in allow:
                    continue
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
                rank += 1
        order = sorted(rrf, key=rrf.get, reverse=True)[:k]
        return [dict(self.chunks[i], score=round(rrf[i], 5)) for i in order]


def normalize(headword):
    """표제어 비교용 정규화: 괄호 속 한자 병기와 공백·기호를 없앤다."""
    return re.sub(r"[\s\^\-·]+", "", re.sub(r"\([^)]*\)", "", headword or ""))


def build(a):
    os.makedirs(a.index_dir, exist_ok=True)
    chunks = make_chunks(a.corpus, a.chunk_chars, a.min_chars)
    with open(os.path.join(a.index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"조각 {len(chunks)}개", flush=True)

    matrix, vocab = build_bm25(chunks, Mecab(), a.bm25_k1, a.bm25_b)
    sparse.save_npz(os.path.join(a.index_dir, "bm25.npz"), matrix)
    json.dump(vocab, open(os.path.join(a.index_dir, "bm25_vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)

    from FlagEmbedding import BGEM3FlagModel
    vecs = encode(BGEM3FlagModel(a.embed_model, use_fp16=True), [c["text"] for c in chunks], a.batch, a.max_length)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, os.path.join(a.index_dir, "dense.faiss"))
    print("색인 완료", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--corpus", default="", help="[build] 사전·백과사전 항목 jsonl")
    ap.add_argument("--chunk-chars", type=int, default=500, help="[build] 본문 조각의 목표 글자 수")
    ap.add_argument("--min-chars", type=int, default=40, help="[build] 이보다 짧은 조각은 버린다")
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--max-length", type=int, default=512, help="임베딩 입력 토큰 상한")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--query", default="", help="[search] 질의")
    ap.add_argument("--k", type=int, default=2, help="[search] 돌려줄 조각 수")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF 상수")
    ap.add_argument("--pool", type=int, default=50, help="검색기마다 뽑을 후보 수")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    else:
        r = Retriever(a.index_dir, a.embed_model, a.rrf_k, a.pool, a.max_length)
        for h in r.search(a.query, a.k):
            print(f"{h['score']:.4f} [{h['source']}·{h['kind']}] {h['text'][:100]}")


if __name__ == "__main__":
    main()
