"""외부 이미지 색인을 만들고 문항 사진과 비슷한 이미지를 찾는다 (논문 4.2절).

DINOv2 임베딩의 코사인 유사도로 검색한다. 임베딩을 정규화해 내적 색인에 넣으므로 내적이 곧 코사인 유사도다.
문항 사진이 색인 이미지를 자르거나 줄인 것이면 유사도가 1에 가깝게 나온다.

  python retrieval/image_index.py build --image-dir ext/images --index-dir work/img_index
  python retrieval/image_index.py query --index-dir work/img_index --manifest ext/manifest.json \
      --data data/test.json --images data/images/test_1280 --out work/img_hits_test.jsonl

manifest.json: {이미지 파일 이름: {"name": 표제어, "desc": 설명, "source": 출처}}
query 출력 행: {question_id, hits: [{name, desc, source, file, score}, ...]}
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class Embedder:
    def __init__(self, model_name, device):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, dtype=self.dtype).to(device).eval()

    def embed(self, images):
        with torch.inference_mode():
            inp = self.proc(images=images, return_tensors="pt")
            inp = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
                   for k, v in inp.items()}
            feats = torch.nn.functional.normalize(self.model(**inp).pooler_output.float(), dim=-1)
        return feats.cpu().numpy()


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))   # 깨진 파일은 빈 이미지로 대신한다


def embed_paths(emb, paths, batch, workers):
    """배치 단위로 임베딩한다. 이미지 복호화는 스레드로 나눠 모델 계산과 겹치게 한다."""
    out = []
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(paths), batch):
            out.append(emb.embed(list(pool.map(load_image, paths[i:i + batch]))))
            if (i // batch) % 20 == 0:
                print(f"  임베딩 {i}/{len(paths)}", flush=True)
    return np.concatenate(out).astype("float32")


def build(a):
    os.makedirs(a.index_dir, exist_ok=True)
    index_path, names_path = os.path.join(a.index_dir, "images.faiss"), os.path.join(a.index_dir, "names.json")
    names = json.load(open(names_path)) if os.path.exists(names_path) else []
    index = faiss.read_index(index_path) if names else None
    known = set(names)
    new = [f for f in sorted(os.listdir(a.image_dir)) if f not in known]
    print(f"새로 임베딩할 이미지 {len(new)}장 (기존 {len(names)}장)", flush=True)
    if new:
        embs = embed_paths(Embedder(a.embed_model, a.device), [os.path.join(a.image_dir, f) for f in new],
                           a.batch, a.workers)
        if index is None:
            index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, index_path)
        json.dump(names + new, open(names_path, "w"))
    print(f"색인 크기 {index.ntotal if index else 0}", flush=True)


def query(a):
    index = faiss.read_index(os.path.join(a.index_dir, "images.faiss"))
    names = json.load(open(os.path.join(a.index_dir, "names.json")))
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    data = json.load(open(a.data, encoding="utf-8"))
    paths = [os.path.join(a.images, r["model_input"]["image_name"]) for r in data]
    embs = embed_paths(Embedder(a.embed_model, a.device), paths, a.batch, a.workers)
    scores, ids = index.search(embs, a.k)
    with open(a.out, "w", encoding="utf-8") as f:
        for rec, ss, ii in zip(data, scores, ids):
            hits, seen = [], set()
            for s, i in zip(ss.tolist(), ii.tolist()):
                if i < 0:
                    continue
                meta = manifest.get(names[i]) or manifest.get(os.path.splitext(names[i])[0]) or {}
                key = meta.get("name") or names[i]
                if key in seen:   # 같은 대상의 다른 사진은 한 번만 남긴다
                    continue
                seen.add(key)
                hits.append({"name": meta.get("name", ""), "desc": meta.get("desc", ""),
                             "source": meta.get("source", ""), "file": names[i], "score": round(s, 4)})
            f.write(json.dumps({"question_id": rec["metadata"]["question_id"], "hits": hits},
                               ensure_ascii=False) + "\n")
    print(f"{len(data)}문항 -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query"])
    ap.add_argument("--index-dir", required=True, help="색인 저장 폴더")
    ap.add_argument("--image-dir", default="", help="[build] 외부 이미지 폴더")
    ap.add_argument("--manifest", default="", help="[query] 이미지별 표제어와 설명")
    ap.add_argument("--data", default="", help="[query] 문항 JSON")
    ap.add_argument("--images", default="", help="[query] 문항 이미지 폴더")
    ap.add_argument("--out", default="", help="[query] 검색 결과 jsonl")
    ap.add_argument("--k", type=int, default=8, help="[query] 문항당 검색할 이미지 수")
    ap.add_argument("--embed-model", default="facebook/dinov2-large")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="이미지 복호화 스레드 수")
    a = ap.parse_args()
    {"build": build, "query": query}[a.cmd](a)


if __name__ == "__main__":
    main()
