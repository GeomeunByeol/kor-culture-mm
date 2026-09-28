"""VLM(Qwen3.8-27B)에 선다형 어댑터(A-MC)를 학습한다 (논문 3.2절).

시각 인코더는 고정하고 언어 모델의 어텐션·MLP 사영에만 LoRA를 붙인다.
--init-adapter를 주면 그 어댑터를 이어서 학습한다(선다형 RAFT).

  python train/train_lora_vlm.py --sft work/sft/vlm_mc.jsonl --images data/images/train_1280 --out adapters/a-mc
  python train/train_lora_vlm.py --sft work/raft/vlm_train_mc.jsonl --images data/images/train_1280 \
      --init-adapter adapters/a-mc --out adapters/raft-mc --epochs 3 --grad-accum 4 --max-len 6144
"""
import argparse
import json
import os
import random

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_KEYS = ("visual", "vision", "merger", "patch")   # 시각 인코더에 속한 모듈 이름의 표지
PAD_ID = 0


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, images_dir, processor, max_len):
        self.rows, self.images_dir, self.proc, self.max_len = rows, images_dir, processor, max_len

    def __len__(self):
        return len(self.rows)

    def messages(self, row, with_answer):
        """학습 행을 채팅 메시지로 바꾼다. 이미지는 첫 user 차례에 넣는다."""
        msgs = []
        for i, m in enumerate(row["messages"]):
            if m["role"] == "assistant" and not with_answer:
                break
            if m["role"] == "user" and row.get("image") and i == 1:
                img = Image.open(os.path.join(self.images_dir, row["image"])).convert("RGB")
                msgs.append({"role": "user", "content": [{"type": "image", "image": img},
                                                         {"type": "text", "text": m["content"]}]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        return msgs

    def __getitem__(self, i):
        row = self.rows[i]
        full = self.proc.apply_chat_template(self.messages(row, True), tokenize=True, add_generation_prompt=False,
                                             return_dict=True, return_tensors="pt")
        prompt = self.proc.apply_chat_template(self.messages(row, False), tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        ids = full["input_ids"][0][: self.max_len]
        labels = ids.clone()
        labels[: prompt["input_ids"].shape[1]] = -100   # 프롬프트 구간은 손실에서 뺀다
        item = {k: v[0] if k in ("input_ids", "attention_mask") else v for k, v in full.items()}
        item.update(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        return item


def collate(batch):
    """글 텐서는 가장 긴 행에 맞춰 오른쪽을 채우고, 이미지 텐서는 0번 축으로 이어 붙인다.

    텐서의 종류는 모양이 아니라 키 이름으로 구분한다. 토큰 길이가 패치 차원과 우연히 같으면
    모양만으로는 둘을 가릴 수 없기 때문이다.
    """
    length = max(b["input_ids"].shape[0] for b in batch)
    out = {}
    for k in batch[0]:
        if any(t in k for t in ("pixel", "grid", "image_sizes")):
            out[k] = torch.cat([b[k] for b in batch], dim=0)
            continue
        fill = {"input_ids": PAD_ID, "labels": -100}.get(k, 0)
        rows = []
        for b in batch:
            t = b[k].reshape(-1) if b[k].dim() > 1 and b[k].shape[0] == 1 else b[k]
            rows.append(torch.nn.functional.pad(t, (0, length - t.shape[-1]), value=fill))
        out[k] = torch.stack(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", help="기반 VLM")
    ap.add_argument("--sft", required=True, help="학습 데이터(messages jsonl, image 필드 포함)")
    ap.add_argument("--images", required=True, help="EXIF 보정된 1,280픽셀 학습 이미지 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", default="", help="이어서 학습할 LoRA 어댑터 폴더")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA 랭크")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA 알파")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA 드롭아웃")
    ap.add_argument("--bs", type=int, default=1, help="배치 크기")
    ap.add_argument("--grad-accum", type=int, default=16, help="기울기 누적 횟수")
    ap.add_argument("--max-len", type=int, default=4096, help="최대 토큰 길이")
    ap.add_argument("--max-pixels", type=int, default=1280 * 1280, help="이미지 한 장의 최대 픽셀 수")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    rows = [json.loads(line) for line in open(a.sft, encoding="utf-8")]
    random.shuffle(rows)
    print(f"학습 예시 {len(rows)}건", flush=True)

    processor = AutoProcessor.from_pretrained(a.model)
    if hasattr(getattr(processor, "image_processor", None), "max_pixels"):
        processor.image_processor.max_pixels = a.max_pixels
    global PAD_ID
    PAD_ID = processor.tokenizer.pad_token_id or 0

    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        # 시각 인코더 아래의 선형층은 LoRA 대상에서 뺀다
        targets = [name for name, mod in model.named_modules()
                   if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGET_LEAVES
                   and not any(k in name for k in VISION_KEYS)]
        model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps = len(rows) * a.epochs / (a.bs * a.grad_accum)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        bf16=True, logging_steps=5, report_to=[], seed=a.seed,
        save_strategy="epoch", save_only_model=True,   # 에폭마다 어댑터를 남긴다
        warmup_steps=max(1, int(a.warmup_ratio * steps)), lr_scheduler_type="cosine",
        remove_unused_columns=False, dataloader_num_workers=4, gradient_checkpointing=True,
    )
    Trainer(model=model, args=args, train_dataset=SFTDataset(rows, a.images, processor, a.max_len),
            data_collator=collate).train()
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=2)
    print(f"어댑터 저장 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
