"""한국어 LLM(A.X-4.0)에 문항 형식별 LoRA 어댑터를 학습한다 (논문 3.2절, 3.3절).

기반 모델의 가중치는 고정하고 어댑터만 학습한다. --init-adapter를 주면 그 어댑터를 이어서 학습하며,
RAFT는 SFT 어댑터를 초깃값으로 이렇게 학습한다.

  # A-SA (단답형)
  python train/train_lora_text.py --sft work/sft/text_sa.jsonl --out adapters/a-sa --lr 5e-5 --load-8bit
  # A-LA (서술형)
  python train/train_lora_text.py --sft work/sft/text_la.jsonl --out adapters/a-la --lr 1e-4 --load-8bit
  # RAFT (단답형): A-SA를 초깃값으로, 참고 자료가 붙은 입력으로 학습
  python train/train_lora_text.py --sft work/raft/text_train_sa.jsonl --init-adapter adapters/a-sa \
      --out adapters/raft-sa --lr 1e-4 --max-len 6144 --bs 1 --grad-accum 4 --load-8bit
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(path, tok, max_len):
    """messages jsonl을 토큰열로 바꾼다. 프롬프트 구간은 손실에서 빼고 정답 구간만 학습한다."""
    examples = []
    for line in open(path, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        target = msgs[-1]["content"].strip() + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[:max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:max_len]
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="skt/A.X-4.0", help="기반 모델")
    ap.add_argument("--sft", required=True, help="학습 데이터(messages jsonl)")
    ap.add_argument("--out", required=True, help="어댑터 저장 폴더")
    ap.add_argument("--init-adapter", default="", help="이어서 학습할 LoRA 어댑터 폴더")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4, help="단답형과 선다형 텍스트 어댑터는 5e-5, 나머지는 1e-4")
    ap.add_argument("--r", type=int, default=16, help="LoRA 랭크")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA 알파")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA 드롭아웃")
    ap.add_argument("--bs", type=int, default=4, help="배치 크기")
    ap.add_argument("--grad-accum", type=int, default=4, help="기울기 누적 횟수")
    ap.add_argument("--max-len", type=int, default=1536, help="최대 토큰 길이. RAFT는 6144")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-8bit", action="store_true", help="기반 모델을 8비트로 올려 학습한다")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = build_examples(a.sft, tok, a.max_len)
    print(f"학습 예시 {len(examples)}건, 평균 {sum(len(e['input_ids']) for e in examples) / len(examples):.0f}토큰",
          flush=True)

    if a.load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, device_map="cuda", dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        bf16=True, logging_steps=5, report_to=[], seed=a.seed,
        save_strategy="epoch", save_only_model=True,   # 에폭마다 어댑터를 남긴다
        lr_scheduler_type="cosine", warmup_steps=a.warmup_steps, weight_decay=0.0,
        gradient_checkpointing=True, remove_unused_columns=False, dataloader_num_workers=2,
    )
    Trainer(model=model, args=args, train_dataset=examples,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=2)
    print(f"어댑터 저장 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
