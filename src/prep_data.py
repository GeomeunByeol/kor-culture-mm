"""대회 배포 zip에서 문항 JSON과 이미지를 풀고, 해상도별 축소 이미지를 만든다.

원본 이미지는 긴 변 중앙값이 약 4,032픽셀이고 평가 이미지의 약 25%가 회전된 채 저장되어 있다.
따라서 EXIF 방향을 먼저 보정한 뒤 축소한다.

  python src/prep_data.py --data-root <zip 폴더> --work data --split train --max-sides 1280,1792
"""
import argparse
import json
import os
import zipfile

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None   # 일부 원본이 Pillow의 픽셀 수 제한을 넘는다


def extract(data_root, split, work, ann_zip):
    """문항 JSON과 원본 이미지를 work 폴더에 푼다. 이미 있으면 건너뛴다."""
    os.makedirs(work, exist_ok=True)
    json_out = os.path.join(work, f"{split}.json")
    if not os.path.exists(json_out):
        zj = zipfile.ZipFile(os.path.join(data_root, ann_zip))
        name = next(n for n in zj.namelist() if split in n)
        open(json_out, "wb").write(zj.read(name))
    raw_dir = os.path.join(work, "images", split)
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        zipfile.ZipFile(os.path.join(data_root, f"{split}.zip")).extractall(raw_dir)
    return json_out, raw_dir


def resize(raw_dir, work, split, max_side, quality):
    """EXIF 방향을 보정하고 긴 변이 max_side를 넘지 않게 줄여 저장한다."""
    dst = os.path.join(work, "images", f"{split}_{max_side}")
    os.makedirs(dst, exist_ok=True)
    made = 0
    for name in sorted(os.listdir(raw_dir)):
        out = os.path.join(dst, name)
        if os.path.exists(out):
            continue
        im = ImageOps.exif_transpose(Image.open(os.path.join(raw_dir, name))).convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        im.save(out, "JPEG", quality=quality)
        made += 1
    print(f"{dst}: 새로 만든 이미지 {made}장")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="대회 배포 zip이 있는 폴더")
    ap.add_argument("--work", default="data", help="풀어 놓을 작업 폴더")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--max-sides", default="1280,1792",
                    help="긴 변 상한(쉼표 구분). 1280은 선다형 VLM, 1792는 관찰문에 쓴다")
    ap.add_argument("--ann-zip", default="korean_submission.zip", help="문항 JSON 세 개가 들어 있는 zip")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    a = ap.parse_args()

    json_path, raw_dir = extract(a.data_root, a.split, a.work, a.ann_zip)
    for side in [int(x) for x in a.max_sides.split(",")]:
        resize(raw_dir, a.work, a.split, side, a.jpeg_quality)
    print("문항 수:", len(json.load(open(json_path, encoding="utf-8"))))


if __name__ == "__main__":
    main()
