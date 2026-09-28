#!/usr/bin/env bash
# VLM(Qwen3.8-27B)을 vLLM으로 구동한다. 관찰문 생성, 확대 판독, 선다형 직접 답에 쓴다.
# 선다형 어댑터(A-MC)와 선다형 RAFT 어댑터를 LoRA로 함께 올린다.
#
#   bash scripts/serve_vlm.sh adapters 8000
set -euo pipefail

ADAPTERS=${1:-adapters}
PORT=${2:-8000}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
MAX_LEN=${MAX_LEN:-16384}
LORA_RANK=${LORA_RANK:-16}

MODULES=()
for name in a-mc raft-mc; do
  [ -f "$ADAPTERS/$name/adapter_config.json" ] && MODULES+=("$name=$ADAPTERS/$name")
done

ARGS=(--port "$PORT" --served-model-name bench --max-model-len "$MAX_LEN"
      --gpu-memory-utilization 0.90 --max-num-seqs 16 --limit-mm-per-prompt '{"image": 1}')
if [ ${#MODULES[@]} -gt 0 ]; then
  ARGS+=(--enable-lora --max-lora-rank "$LORA_RANK" --lora-modules "${MODULES[@]}")
fi
vllm serve "$MODEL" "${ARGS[@]}"
