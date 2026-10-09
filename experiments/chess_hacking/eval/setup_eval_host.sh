#!/usr/bin/env bash
# Idempotent setup of the ctfish evaluation host: a Linux box with Docker
# (the sandboxes) and Python 3.10+. No GPU: the model is served elsewhere and
# reached over an SSH tunnel (ssh -N -L 8000:127.0.0.1:8000 <gpu-host>).
#
#   bash eval/setup_eval_host.sh [--rebuild-image]
#
# Builds the repository venv (../../.venv, see the top-level README), installs
# the vendored ctfish task and `steered` provider, and prebuilds the sandbox
# image ctfish-env:latest (python 3.11 + stockfish 17-1 + python-chess 1.11.1).
# Containers run with network_mode: none, so the image must exist in advance.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
VENV=${VENV:-$ROOT/.venv}
CTFISH_IMAGE=${CTFISH_IMAGE:-ctfish-env:latest}
REBUILD=0; [ "${1:-}" = "--rebuild-image" ] && REBUILD=1

command -v docker >/dev/null || { echo "docker missing" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable" >&2; exit 1; }

echo "=== venv $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -r "$ROOT/requirements.txt"
"$VENV/bin/python" -m pip install -q -e "$HERE/vendor"

echo "=== sandbox image $CTFISH_IMAGE"
if [ "$REBUILD" = 1 ] || ! docker image inspect "$CTFISH_IMAGE" >/dev/null 2>&1; then
  docker build -t "$CTFISH_IMAGE" --build-arg "STOCKFISH_VERSION=${CTFISH_STOCKFISH_VERSION:-17-1}" "$HERE/vendor/ctfish/env"
else
  echo "image present (--rebuild-image to force)"
fi

echo "=== verification"
"$VENV/bin/python" - <<'PY'
import inspect_ai, anthropic, openai
from ctfish import ctfish
import steered_provider
print("inspect_ai", inspect_ai.__version__, "| anthropic", anthropic.__version__, "| openai", openai.__version__)
print("ctfish", ctfish, "| steered_provider", steered_provider.__file__)
PY
docker run --rm --network none "$CTFISH_IMAGE" sh -c 'python -c "import chess; print(\"chess\", chess.__version__)"; command -v stockfish'
echo "done. Plumbing test (no keys, no model): $VENV/bin/python eval/ctfish_smoke.py --mode scripted"
