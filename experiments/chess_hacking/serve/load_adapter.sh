#!/usr/bin/env bash
# load_adapter.sh <name> <path> [port]
set -uo pipefail

NAME=${1:?usage: load_adapter.sh <name> <path> [port]}
APATH=${2:?usage: load_adapter.sh <name> <path> [port]}
PORT=${3:-8000}
HOST=${HOST:-127.0.0.1}
BASE="http://$HOST:$PORT"
TIMEOUT=${TIMEOUT:-900}

have() { curl -fsS --max-time 15 "$BASE/v1/models" |
         python3 -c 'import json,sys;ids=[m["id"] for m in json.load(sys.stdin)["data"]];sys.exit(0 if sys.argv[1] in ids else 1)' "$1"; }

curl -fsS --max-time 15 "$BASE/v1/models" >/dev/null 2>&1 || {
  echo "no vLLM answering /v1/models at $BASE" >&2; exit 1; }

if have "$NAME"; then echo "adapter '$NAME' already loaded on $BASE"; exit 0; fi

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

if have "$NAME"; then echo "OK: '$NAME' is in /v1/models on $BASE"; exit 0; fi
echo "FAILED: '$NAME' not in /v1/models after load" >&2; exit 1
