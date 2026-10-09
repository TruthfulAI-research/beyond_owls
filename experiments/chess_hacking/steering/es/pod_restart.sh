#!/bin/bash
# Kill the running vLLM server and start the ES one. RUNS ON THE POD.
# Exits non-zero if the old server will not die: a second server would bind a
# busy port, fail, and leave the OLD vectors being served on :8000 -- which is
# the exact failure the manifest check downstream exists to catch, but it is
# cheaper to refuse here.
set -uo pipefail
SERVE_SCRIPT="$1"
LOG="$2"

pkill -f '[a]pi_server'
sleep 3
pkill -f '[V]LLM::EngineCore' 2>/dev/null
for i in $(seq 1 60); do
  pgrep -f '[a]pi_server' >/dev/null || { echo "old server gone after ${i}s"; break; }
  sleep 1
done
if pgrep -f '[a]pi_server' >/dev/null; then
  echo "old server still alive after 60s, SIGKILL"
  pkill -9 -f '[a]pi_server'
  pkill -9 -f '[V]LLM::EngineCore' 2>/dev/null
  sleep 5
fi
if pgrep -f '[a]pi_server' >/dev/null; then
  echo "FATAL: could not kill the old server (pids: $(pgrep -f '[a]pi_server' | tr '\n' ' '))"
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
[ -f "$LOG" ] && mv "$LOG" "$LOG.prev-$(date +%s)"
setsid nohup "$SERVE_SCRIPT" < /dev/null > "$LOG" 2>&1 &
echo "launched $SERVE_SCRIPT -> $LOG"
sleep 3
pgrep -f '[a]pi_server' >/dev/null && echo "api_server pid: $(pgrep -f '[a]pi_server' | tr '\n' ' ')" || echo "WARNING: no api_server pid yet (still importing)"
exit 0
