#!/usr/bin/env bash
# load_adapter.sh <name> <path> [port]
#
#   bash serve/load_adapter.sh legal_s0_ep10 /abs/path/runs/legal_s0/epoch_09_vllm
#
# Registers a LoRA adapter with an already-running vLLM server that was started
# with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True. Idempotent: if <name> is already
# in /v1/models it returns 0 without touching the server.
#
# Timing: a 934 MB fp32 r=32 adapter takes about 3 minutes to load.
# curl's default has no timeout, but we set an explicit one so a wedged server
# fails loudly instead of hanging a launch script forever.
set -uo pipefail

NAME=${1:?usage: load_adapter.sh <name> <path> [port]}
APATH=${2:?usage: load_adapter.sh <name> <path> [port]}
PORT=${3:-8000}
HOST=${HOST:-127.0.0.1}
BASE="http://$HOST:$PORT"
TIMEOUT=${TIMEOUT:-900}

have() { curl -fsS --max-time 15 "$BASE/v1/models" |
         python3 -c 'import json,sys;ids=[m["id"] for m in json.load(sys.stdin)["data"]];sys.exit(0 if sys.argv[1] in ids else 1)' "$1"; }

# Validate via /v1/models, not /health: RunPod's nginx answers /health with
# 200 + HTML, and so would any other reverse proxy sitting on a stray port.
curl -fsS --max-time 15 "$BASE/v1/models" >/dev/null 2>&1 || {
  echo "no vLLM answering /v1/models at $BASE" >&2; exit 1; }

if have "$NAME"; then echo "adapter '$NAME' already loaded on $BASE"; exit 0; fi

# Path is resolved by the SERVER process, so check it where the server runs.
[ -f "$APATH/adapter_model.safetensors" ] || \
  echo "WARN: $APATH/adapter_model.safetensors not visible from here (fine if the server is remote)" >&2

echo "loading '$NAME' from $APATH on $BASE (up to ${TIMEOUT}s) ..."
t0=$(date +%s)
body=$(curl -sS --max-time "$TIMEOUT" -w '\n%{http_code}' \
   -X POST "$BASE/v1/load_lora_adapter" \
   -H 'Content-Type: application/json' \
   -d "$(python3 -c 'import json,sys;print(json.dumps({"lora_name":sys.argv[1],"lora_path":sys.argv[2]}))' "$NAME" "$APATH")")
code=$(printf '%s' "$body" | tail -1)
echo "  http $code after $(( $(date +%s) - t0 ))s: $(printf '%s' "$body" | head -n -1 | head -c 300)"

# Trust /v1/models, not the status code: confirm the adapter is actually servable.
if have "$NAME"; then echo "OK: '$NAME' is in /v1/models on $BASE"; exit 0; fi
echo "FAILED: '$NAME' not in /v1/models after load" >&2; exit 1
