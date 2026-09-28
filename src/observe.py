"""릴레이 1단계: VLM이 질문에 맞춰 이미지를 관찰문으로 옮긴다 (논문 3.1절, o = V(I, q)).

VLM은 답을 내지 않고 질문과 관련된 시각적 사실(글자 원문, 형태, 색, 재질, 개수, 장소 단서)만 적는다.
지식과 추론은 2단계의 LLM이 맡는다.

  python src/observe.py --data data/test.json --images data/images/test_1792 --out work/obs_test.jsonl

출력 행: {question_id, form, obs, sec, error}
"""
import argparse
import json
import os
import time

from common import (add_server_args, chat_text, done_ids, image_b64, image_message, load_records,
                    run_parallel, sanitize)

OBS_SYSTEM = (
    "당신은 이미지를 정밀하게 관찰하여 기록하는 조사원입니다. 답을 추측하거나 결론을 내리지 말고, "
    "보이는 사실만 한국어로 빠짐없이 적으십시오."
)

OBS_INSTR = (
    "아래 질문에 답하는 데 필요한 정보를 이미지에서 찾아 기록하시오. 답 자체는 쓰지 마시오.\n"
    "포함할 것:\n"
    "1) 질문과 관련된 글자·간판·표지·라벨은 원문 그대로. 단, 같은 종류의 글자(전화번호 목록, 반복되는 자모·숫자 등)는 "
    "대표 1~2개만 적고 '외 다수'로 줄이시오. 같은 내용을 반복해서 쓰지 마시오.\n"
    "2) 질문이 가리키는 물체·인물·복식·음식·건물·도구의 구체적 형태, 색, 재질, 무늬, 부위별 특징, 개수\n"
    "3) 장소·시대·행사·계절을 짐작하게 하는 단서 (배경, 소품, 복장, 현수막 등)\n"
    "4) 선택지가 있다면 각 선택지를 판별하는 데 필요한 시각적 근거\n"
    "형식: 번호 붙인 항목 목록, 항목당 한두 문장. 불확실한 것은 '~로 보임'이라고 표시. 전체 300자 내외, 최대 500자."
)


def obs_prompt(rec):
    mi = rec["model_input"]
    parts = ["[질문]", mi["question"]]
    if mi.get("options"):
        parts.append("[선택지]\n" + "\n".join(mi["options"]))
    parts.append(OBS_INSTR)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="대회 문항 JSON")
    ap.add_argument("--images", required=True, help="EXIF 보정된 1,792픽셀 이미지 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="greedy로 '모두 나열'을 시키면 전화번호나 자모를 끝없이 반복하므로 약한 벌점을 준다")
    ap.add_argument("--limit", type=int, default=0)
    add_server_args(ap)
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs)

    def one(rec):
        st = time.time()
        img = os.path.join(a.images, rec["model_input"]["image_name"])
        payload = {"model": a.model, "temperature": 0.0, "top_p": 1.0, "max_tokens": a.max_tokens,
                   "repetition_penalty": a.repetition_penalty,
                   "messages": [{"role": "system", "content": OBS_SYSTEM},
                                {"role": "user", "content": image_message(image_b64(img), obs_prompt(rec))}]}
        if ctk:
            payload["chat_template_kwargs"] = ctk
        obs, err = chat_text(a.url, payload, a.timeout)
        return {"question_id": rec["metadata"]["question_id"], "form": rec["metadata"]["question_form"],
                "obs": sanitize(obs), "sec": round(time.time() - st, 2), "error": err}

    done = done_ids(a.out)
    todo = [r for r in load_records(a.data, limit=a.limit) if r["metadata"]["question_id"] not in done]
    print(f"관찰 대상 {len(todo)}건 (완료 {len(done)}건)", flush=True)
    run_parallel(todo, one, a.out, a.workers)


if __name__ == "__main__":
    main()
