#!/usr/bin/env bash
# Launch a plain vLLM server with runtime LoRA loading (the students), detached.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3.6-27B}
HOST=${HOST:-127.0.0.1} PORT=${PORT:-8000}
export HF_HOME=${HF_HOME:-$HERE/hf}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export PYTHONUNBUFFERED=1
unset VLLM_PLUGINS STEER_ENABLE STEER_VECTOR_DIR STEER_VECTORS PYTHONPATH
LOG=${LOG:-$HERE/serve_lora.log}

if curl -fsS -o /dev/null "http://$HOST:$PORT/v1/models" 2>/dev/null; then
  echo "a server is already up on $HOST:$PORT"; exit 0
fi
CMD=(python -u -m vllm.entrypoints.openai.api_server
     --model "$MODEL" --served-model-name "$MODEL"
     --host "$HOST" --port "$PORT" --dtype bfloat16
     --max-model-len "${MAX_MODEL_LEN:-65536}" --max-num-seqs "${MAX_NUM_SEQS:-64}"
     --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
     --reasoning-parser qwen3 --enable-prefix-caching
     --enable-lora --max-lora-rank 32 --max-loras "${MAX_LORAS:-4}" --max-cpu-loras 16
     --no-enable-log-requests "$@")
echo "cmd: ${CMD[*]}"
setsid nohup "${CMD[@]}" < /dev/null >> "$LOG" 2>&1 &
echo "started pid $!, log $LOG; wait for 'Application startup complete'"
