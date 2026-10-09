#!/usr/bin/env bash
# Launch a vLLM server with per-request activation steering.
#
#   MODEL=/models/my-model STEER_VECTOR_DIR=/vectors pod/scripts/serve.sh
#
# Any extra arguments are appended to the vLLM command line. See pod/README.md
# for the full set of variables.
set -euo pipefail

# beyond_owls: this script sits next to sitecustomize.py (vendor/), not in pod/scripts/.
pod_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:?MODEL must be set}"
: "${STEER_VECTOR_DIR:?STEER_VECTOR_DIR must be set}"

export STEER_ENABLE=1
export STEER_VECTOR_DIR
export STEER_VECTOR_ARG=${STEER_VECTOR_ARG:-steer_vector}
export STEER_ARG=${STEER_ARG:-steer_strength}
export STEER_DEBUG=${STEER_DEBUG:-0}
if [ -n "${STEER_VECTORS:-}" ]; then export STEER_VECTORS; fi

# sitecustomize.py lives here and is imported by every process that inherits
# PYTHONPATH, including the engine and worker processes vLLM spawns.
export PYTHONPATH=$pod_dir${PYTHONPATH:+:$PYTHONPATH}

# Endpoint plugins add HTTP routes, so vLLM refuses to load one unless it is
# named here — an unset VLLM_PLUGINS means the manifest endpoint is silently
# absent. Note that this variable is also the allowlist for vLLM's other plugin
# groups: if this model needs a general plugin, name it here too.
export VLLM_PLUGINS=${VLLM_PLUGINS:-steering}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

# CUDA graphs are left on. The steering is inside the target block's forward
# (see patch.py), so capture records it like any other arithmetic in the model.
# Measured on Qwen3.6-27B on one H200, that is worth about 3x the output rate of
# a single request against the same server started eager.
#
# VLLM_DISABLE_COMPILE_CACHE is set above because vLLM's cache key does not
# include this plugin: without it a steered engine can replay an artefact
# compiled without the steering, which is unsteered generation reporting
# success.
#
# --middleware is what turns a request naming a vector this server does not
# have into a 400. Without it such a request is served by the unsteered model,
# and the only sign of it is a line in the worker's log.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "$MODEL")}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --middleware vllm_steering.middleware.SteeringValidation \
  "$@"
