#!/usr/bin/env bash
set -euo pipefail

PORT=${1:-15000}
LOG=/tmp/mkdocs_serve.log

echo "Building docs..."
uv run mkdocs build

echo "Serving at http://0.0.0.0:${PORT} (log: ${LOG})"
nohup uv run mkdocs serve --dev-addr "0.0.0.0:${PORT}" > "${LOG}" 2>&1 &
echo "PID $!"
