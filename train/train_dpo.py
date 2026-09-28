"""SFT 어댑터에 직접 선호 최적화(DPO)를 이어서 적용한다 (논문 3.5절).

SFT 어댑터를 정책으로 삼아 이어서 학습하고, 참조 모델은 학습 전 어댑터의 고정 사본이다(DPOTrainer가 만든다).
프롬프트는 추론 때와 같은 문자열로 만든다.

  python train/train_dpo.py --form LA --pairs work/dpo/pairs_la.jsonl --adapter adapters/a-la \
      --out adapters/dpo-la --max-len 2048
  python train/train_dpo.py --form SA --pairs work/dpo/pairs_sa.jsonl --adapter adapters/a-sa \
      --out adapters/dpo-sa --max-len 1024
"""
import argparse
import dataclasses
import json
import os
import sys

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import TXT_SYSTEM, build_text_prompt, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["SA", "LA"], required=True)
    ap.add_argument("--pairs", required=True, help="build_dpo_pairs.py 출력")
    ap.add_argument("--base", default="skt/A.X-4.0", help="기반 모델")
    ap.add_argument("--adapter", required=True, help="이어서 학습할 SFT 어댑터")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.5, help="DPO 베타")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bs", type=int, default=1, help="배치 크기")
    ap.add_argument("--grad-accum", type=int, default=8, help="기울기 누적 횟수. 유효 배치 크기는 bs x grad-accum")
    ap.add_argument("--max-len", type=int, default=1024, help="최대 토큰 길이. 서술형은 2048")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.adapter)

    def prompt(row):
        # build_text_prompt는 문항 레코드를 받으므로 선호 쌍을 같은 모양으로 감싼다
        rec = {"metadata": {"question_form": a.form}, "model_input": {"question": row["question"]}}
        user = build_text_prompt(rec, row["observation"], row.get("refs", ""))
        return tok.apply_chat_template([{"role": "system", "content": TXT_SYSTEM}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    ds = Dataset.from_list([{"prompt": prompt(r), "chosen": r["chosen"].strip(), "rejected": r["rejected"].strip()}
                            for r in read_jsonl(a.pairs)])
    lens = [len(tok(p + c).input_ids) for p, c in zip(ds["prompt"], ds["chosen"])]
    print(f"선호 쌍 {len(ds)}개, 최대 {max(lens)}토큰 (상한 {a.max_len}, 초과 {sum(n > a.max_len for n in lens)}개)",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                 dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True, adapter_name="default")

    kwargs = dict(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, beta=a.beta, seed=a.seed,
                  per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum, max_length=a.max_len,
                  gradient_checkpointing=True, bf16=True, logging_steps=5, save_strategy="epoch", report_to=[])
    # trl 버전에 따라 없는 인자는 뺀다
    known = {f.name for f in dataclasses.fields(DPOConfig)}
    trainer = DPOTrainer(model=model, args=DPOConfig(**{k: v for k, v in kwargs.items() if k in known}),
                         train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(a.out, selected_adapters=["default"])
    tok.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=2)
    print(f"DPO 어댑터 저장 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
