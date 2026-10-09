#!/usr/bin/env bash
# One-time setup of the GPU host (one H100/H200 class GPU, CUDA 13 capable driver).
#
#   bash serve/pod_setup.sh            # serving venv + training venv + model download
#
# Serving venv  ($VENV):       vllm==0.27.1 (pins torch 2.13 cu130), the vendored
#                              steering plugin and vector format, the generation client.
# Training venv ($VENV_TRAIN): torch, transformers 5.16.1, peft 0.20.0 -- the
#                              versions the paper's students were trained with.
# Both are created with uv. Python 3.12 is required by the vLLM plugin.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HERE/.venv-serve}
VENV_TRAIN=${VENV_TRAIN:-$HERE/.venv-train}
MODEL=${MODEL:-Qwen/Qwen3.6-27B}
export HF_HOME=${HF_HOME:-$HERE/hf}

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

echo "=== serving venv: $VENV"
[ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
uv pip install -p "$VENV/bin/python" "vllm==0.27.1" openai numpy transformers "huggingface_hub[hf_transfer]"
uv pip install -p "$VENV/bin/python" --no-deps -e "$HERE/vendor"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import entry_points
eps = {e.name: e.value for e in entry_points(group="vllm.endpoint_plugins")}
assert eps.get("steering") == "vllm_steering.endpoint:SteeringEndpointPlugin", eps
import vllm, vllm_steering, steering_vectors
print("vllm", vllm.__version__, "| plugin", vllm_steering.__file__)
PY

echo "=== training venv: $VENV_TRAIN"
[ -x "$VENV_TRAIN/bin/python" ] || uv venv -p 3.12 "$VENV_TRAIN"
uv pip install -p "$VENV_TRAIN/bin/python" torch "transformers==5.16.1" "peft==0.20.0" safetensors accelerate numpy
"$VENV_TRAIN/bin/python" -c "import torch, transformers, peft; print('torch', torch.__version__, torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__)"

echo "=== model: $MODEL -> $HF_HOME"
"$VENV/bin/python" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
echo "done. Serve the teacher with: $VENV/bin/activate && bash serve/serve_steered.sh"
