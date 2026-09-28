#!/usr/bin/env bash
# 한국어 LLM(A.X-4.0)을 vLLM으로 FP8 구동하고, 문항 형식별 LoRA 어댑터를 함께 올린다.
# 어댑터는 요청의 model 이름으로 바꿔 쓰므로 어댑터마다 서버를 따로 띄우지 않는다.
#
#   bash scripts/serve_llm.sh adapters 8000
set -euo pipefail

ADAPTERS=${1:-adapters}   # 어댑터 폴더
PORT=${2:-8000}
MODEL=${MODEL:-skt/A.X-4.0}
MAX_LEN=${MAX_LEN:-8192}
LORA_RANK=${LORA_RANK:-16}

# 폴더에 있는 어댑터를 모두 "이름=경로" 꼴로 등록한다
MODULES=()
for d in "$ADAPTERS"/*/; do
  [ -f "$d/adapter_config.json" ] || continue
  grep -q "A.X" "$d/adapter_config.json" || continue   # VLM용 어댑터는 뺀다
  MODULES+=("$(basename "$d")=$d")
done

ARGS=(--port "$PORT" --served-model-name bench --quantization fp8 --max-model-len "$MAX_LEN"
      --gpu-memory-utilization 0.90 --max-num-seqs 32)
if [ ${#MODULES[@]} -gt 0 ]; then
  ARGS+=(--enable-lora --max-lora-rank "$LORA_RANK" --lora-modules "${MODULES[@]}")
fi
vllm serve "$MODEL" "${ARGS[@]}"
