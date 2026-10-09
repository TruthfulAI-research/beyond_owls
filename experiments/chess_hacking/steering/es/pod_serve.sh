#!/bin/bash
# Steered vLLM server for the ES run. RUNS ON THE POD; written by es/deploy.py with
# the placeholders filled in. STEER_VECTOR_DIR is the candidate directory, so the
# served set is exactly the candidates of the current iteration.
export HF_HOME=/workspace/hf
export MODEL=Qwen/Qwen3.6-27B SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
export STEER_VECTOR_DIR=__VECTOR_DIR__
export MAX_MODEL_LEN=__MAX_MODEL_LEN__ MAX_NUM_SEQS=__MAX_NUM_SEQS__ GPU_MEMORY_UTILIZATION=__GPU_UTIL__
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
unset STEER_VECTORS
# prefix caching pinned OFF: the cache is keyed on token ids only, so it would
# serve KV computed under a DIFFERENT steering vector.
exec bash __CHESS_DIR__/vendor/serve_vllm_steering.sh --reasoning-parser qwen3 --no-enable-prefix-caching
