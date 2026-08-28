#!/bin/bash
# Start/stop the Training Observatory as a detached daemon (survives SSH/Claude sessions).
#
#   ./viewer/serve.sh start|stop|restart|status
#
# Environment overrides:
#   HOST=0.0.0.0            bind address (default 127.0.0.1 -> needs an SSH tunnel)
#   PORT=8090               port
#   TOKENIZER=hamishivi/Qwen3.5-9B   tokenizer for trace decoding
#   ENV_REPO=<path>         repo whose synced uv env to run in (default: this repo;
#                           set when running from a worktree without a synced venv)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_REPO="${ENV_REPO:-$REPO}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
TOKENIZER="${TOKENIZER:-hamishivi/Qwen3.5-9B}"
RUN_DIR="$REPO/.viewer_cache"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
  if running; then
    echo "already running (pid $(cat "$PID_FILE")) -> http://$HOST:$PORT"
    return 0
  fi
  mkdir -p "$RUN_DIR"
  cd "$ENV_REPO"
  PYTHONPATH="$REPO" setsid nohup uv run --no-sync python -u -m viewer.server \
    --registry "$REPO/viewer/registry" \
    --rollouts-dir "$REPO/rl_rollouts" \
    --tokenizer "$TOKENIZER" \
    --host "$HOST" --port "$PORT" --verbose \
    >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"ok"'; then
      echo "started (pid $(cat "$PID_FILE")) -> http://$HOST:$PORT   log: $LOG_FILE"
      return 0
    fi
    sleep 1
  done
  echo "did not become healthy within 60s; check $LOG_FILE" >&2
  return 1
}

stop() {
  if running; then
    local pid; pid="$(cat "$PID_FILE")"
    # uv run wraps the python process; kill the whole detached process group.
    kill -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped"
  else
    rm -f "$PID_FILE"
    echo "not running"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status)
    if running; then
      echo "running (pid $(cat "$PID_FILE")) -> http://$HOST:$PORT"
      curl -s -m 3 "http://127.0.0.1:$PORT/api/health" && echo
    else
      echo "not running"
    fi
    ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
