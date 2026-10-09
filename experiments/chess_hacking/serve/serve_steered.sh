#!/usr/bin/env bash
# Launch the steered vLLM server (the teacher) on the GPU host, detached.
#
#   bash serve/serve_steered.sh                      # serves steering/vectors (9012, 9077-9079)
#   RETURN_TOKEN_IDS=1 bash serve/serve_steered.sh   # token ids in logprobs (gen/generate.py default)
#   bash serve/serve_steered.sh --foreground
#
# Settings pinned to the paper's teacher serving:
#   --max-logprobs 32                 the generator asks for top_logprobs=32
#   --logprobs-mode raw_logprobs      the model's own distribution, before top_k
#   --no-enable-prefix-caching        the prefix cache is keyed on token ids only
#                                     and would reuse KV computed under another vector
#   --reasoning-parser qwen3          thinking and answer come back separately
# The server binds loopback only; reach it over an SSH tunnel.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export MODEL=${MODEL:-Qwen/Qwen3.6-27B}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
export STEER_VECTOR_DIR=${STEER_VECTOR_DIR:-$HERE/steering/vectors}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
export HOST=${HOST:-127.0.0.1} PORT=${PORT:-8000}
export HF_HOME=${HF_HOME:-$HERE/hf}
# vLLM's compile cache key does not include the plugin: never replay an artefact
# compiled without the steering. FlashInfer's JIT sampler does not build on the
# pinned stack; vLLM's native torch sampler draws from the same distribution.
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export PYTHONUNBUFFERED=1
unset STEER_VECTORS   # serve the whole directory
LOG=${LOG:-$HERE/serve_steered.log}

FOREGROUND=0
[ "${1:-}" = "--foreground" ] && { FOREGROUND=1; shift; }
ls -d "$STEER_VECTOR_DIR"/[0-9][0-9][0-9][0-9] >/dev/null 2>&1 \
  || { echo "no vector directories in $STEER_VECTOR_DIR: run steering/build_vector.py first" >&2; exit 1; }
if curl -fsS -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; then
  echo "a server is already up on $HOST:$PORT"; exit 0
fi

EXTRA=()
[ "${RETURN_TOKEN_IDS:-0}" = "1" ] && EXTRA+=(--return-tokens-as-token-ids)
CMD=(bash "$HERE/vendor/serve_vllm_steering.sh"
     --reasoning-parser qwen3
     --no-enable-prefix-caching
     --max-logprobs 32 --logprobs-mode raw_logprobs
     --no-enable-log-requests
     "${EXTRA[@]}" "$@")
echo "cmd: ${CMD[*]}"
if [ "$FOREGROUND" = 1 ]; then exec "${CMD[@]}"; fi
setsid nohup "${CMD[@]}" < /dev/null >> "$LOG" 2>&1 &
echo "started pid $!, log $LOG; wait for 'Application startup complete', then"
echo "  curl -s http://$HOST:$PORT/steering/vectors"
