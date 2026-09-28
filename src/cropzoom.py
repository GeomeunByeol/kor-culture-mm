"""불확실한 글자 영역을 확대해서 다시 읽고 관찰문에 덧붙인다 (논문 4.2절의 확대 판독).

1) 확대 대상 선정: 관찰문이 글자를 읽지 못했다고 밝혔거나 글자 항목에 추정 표현이 있고,
   원본 해상도가 관찰 해상도보다 충분히 클 때만 고른다.
2) VLM에게 질문과 관련된 글자 영역의 경계 상자를 묻는다.
3) 원본 이미지에서 그 영역을 잘라 VLM에게 글자를 그대로 옮겨 적게 한다.
4) 관찰문 끝에 "[확대 판독]" 블록을 붙인다.

  python src/cropzoom.py --data data/test.json --obs work/obs_test.jsonl \
      --cache data/images/test_1792 --orig data/images/test --out work/obs_test_zoom.jsonl

대상이 아닌 행도 그대로 복사하므로 출력 파일을 다음 단계의 --obs로 바로 쓸 수 있다.
"""
import argparse
import base64
import json
import os
import re
from io import BytesIO

from PIL import Image, ImageOps

from common import add_server_args, chat_text, image_b64, image_message, load_records, read_jsonl, write_jsonl

Image.MAX_IMAGE_PIXELS = None

SYS = "당신은 사진 속 글자를 정확히 읽는 판독 전문가입니다."
BOX_PROMPT = ("아래 질문에 답하는 데 필요한 글자(간판·표지판·라벨·안내문 등)가 사진의 어느 영역에 있는지 찾으시오.\n"
              "[질문]\n{q}\n\n"
              "그 영역의 경계 상자를 사진 전체 너비·높이에 대한 비율(0~1)로, 다른 말 없이 JSON 한 줄로만 출력하시오: "
              '{{"x1":0.12,"y1":0.30,"x2":0.55,"y2":0.48}}. 해당 글자가 없으면 {{"none":true}}.')
READ_PROMPT = ("이 사진은 원본을 확대한 부분입니다. 보이는 글자를 원문 그대로, 줄 단위로 옮겨 적으시오. "
               "숫자·단위·괄호·띄어쓰기를 그대로 유지하고, 읽을 수 없는 글자는 '□'로 표시하시오. "
               "설명이나 해석은 쓰지 말고 글자만 적으시오. 글자가 전혀 없으면 '[글자 없음]'이라고 쓰시오.")

# 관찰문이 스스로 읽지 못했다고 밝힌 표현
ADMITTED = re.compile(r"식별\s*불가|판독\s*불[가능]|판독 불확실|흐려|흐릿|희미|너무 작아|작아서|"
                      r"해상도로 인해|알아보기 어려|정확히 읽을 수 없|읽을 수 없")
# 글자를 다루는 항목과 그 안의 추정 표현
TEXT_NOUN = re.compile(r"글자|글씨|문구|표기|텍스트|문자|라벨|간판|안내판|안내문|현수막|숫자|"
                       r"제목|명칭|상호|가격|한자|영문")
HEDGE = re.compile(r"추정|유사|보임|보인다|보이며|~?로 보|듯|가능성|불확실")


def needs_zoom(obs, orig_path, obs_side, min_ratio):
    """확대 대상인지 판정한다. 해상도 여유가 없으면 흐린 그림을 키울 뿐이므로 제외한다."""
    w, h = Image.open(orig_path).size
    if max(w, h) / obs_side < min_ratio:
        return False
    if ADMITTED.search(obs):
        return True
    items = [x for x in re.split(r"(?:^|\s)\d+\)\s*", obs) if x.strip()]
    return any(TEXT_NOUN.search(it) and HEDGE.search(it) for it in items)


def parse_box(raw):
    """VLM이 출력한 경계 상자 JSON을 읽는다. 형식이 틀리면 None을 돌려준다."""
    m = re.search(r"\{[^{}]*\}", raw)
    if not m:
        return None
    try:
        d = {k.replace("_", "").lower(): v for k, v in json.loads(m.group(0)).items()}
        if d.get("none"):
            return None
        box = tuple(float(d[k]) for k in ("x1", "y1", "x2", "y2"))
    except Exception:
        return None
    x1, y1, x2, y2 = box
    return box if (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) else None


def ask(a, ctk, b64, text, max_tokens):
    payload = {"model": a.model, "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
               "repetition_penalty": a.repetition_penalty,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": image_message(b64, text)}]}
    if ctk:
        payload["chat_template_kwargs"] = ctk
    return chat_text(a.url, payload, a.timeout)[0].strip()


def zoom_one(rec, a, ctk):
    """한 문항을 확대 판독한다. 성공하면 관찰문에 붙일 블록을, 실패하면 None을 돌려준다."""
    name = rec["model_input"]["image_name"]
    raw = ask(a, ctk, image_b64(os.path.join(a.cache, name)),
              BOX_PROMPT.format(q=rec["model_input"]["question"]), 60)
    box = parse_box(raw)
    if box is None:
        return None
    im = ImageOps.exif_transpose(Image.open(os.path.join(a.orig, name)))
    W, H = im.size
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * a.pad, (y2 - y1) * a.pad
    crop = im.crop((int(max(0, x1 - pw) * W), int(max(0, y1 - ph) * H),
                    int(min(1, x2 + pw) * W), int(min(1, y2 + ph) * H)))
    if min(crop.size) < 32:
        return None
    # 작은 표지판은 글자가 읽히도록 키운다
    if max(crop.size) < a.min_crop_side:
        f = min(a.max_upscale, a.min_crop_side / max(crop.size))
        crop = crop.resize((int(crop.size[0] * f), int(crop.size[1] * f)), Image.LANCZOS)
    scale = max(crop.size) / max(1, (x2 - x1) * a.side)
    crop.thumbnail((a.side, a.side))
    buf = BytesIO()
    crop.convert("RGB").save(buf, "JPEG", quality=90)
    text = ask(a, ctk, base64.b64encode(buf.getvalue()).decode(), READ_PROMPT, 300)
    if not text or "[글자 없음]" in text:
        return None
    return f"[확대 판독] (원본에서 글자 영역을 {scale:.1f}배 확대해 다시 읽음)\n{text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs", required=True, help="observe.py 출력")
    ap.add_argument("--cache", required=True, help="관찰에 쓴 1,792픽셀 이미지 폴더")
    ap.add_argument("--orig", required=True, help="원본 이미지 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="SA", help="확대 판독을 적용할 문항 형식(쉼표 구분)")
    ap.add_argument("--side", type=int, default=1792, help="관찰 해상도(긴 변)")
    ap.add_argument("--min-ratio", type=float, default=1.5, help="원본 긴 변 / 관찰 해상도의 하한")
    ap.add_argument("--pad", type=float, default=0.2, help="경계 상자 둘레에 더할 여백 비율")
    ap.add_argument("--min-crop-side", type=int, default=600, help="이보다 작은 조각은 키운다")
    ap.add_argument("--max-upscale", type=float, default=3.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    add_server_args(ap)
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)
    forms = set(a.forms.split(","))
    recs = {r["metadata"]["question_id"]: r for r in load_records(a.data)}

    rows, n_target, n_done = read_jsonl(a.obs), 0, 0
    for row in rows:
        rec = recs[row["question_id"]]
        orig = os.path.join(a.orig, rec["model_input"]["image_name"])
        if rec["metadata"]["question_form"] not in forms or not needs_zoom(row["obs"], orig, a.side, a.min_ratio):
            continue
        n_target += 1
        block = zoom_one(rec, a, ctk)
        row["zoom"] = bool(block)
        if block:
            row["obs"] = row["obs"].rstrip() + "\n\n" + block
            n_done += 1
    write_jsonl(a.out, rows)
    print(f"확대 대상 {n_target}건 중 {n_done}건 판독 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
