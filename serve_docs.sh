#!/usr/bin/env bash
set -euo pipefail

PORT=${1:-15000}
LOG=/tmp/mkdocs_serve.log

EXISTING=$(pgrep -f "mkdocs serve" || true)
if [ -n "${EXISTING}" ]; then
    read -r -p "Existing mkdocs server found (PIDs: ${EXISTING}). Kill it? [y/N] " REPLY
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        kill ${EXISTING}
        echo "Killed."
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "Building docs..."
uv run mkdocs build

echo "Serving at http://0.0.0.0:${PORT} (log: ${LOG})"
nohup uv run mkdocs serve --dev-addr "0.0.0.0:${PORT}" > "${LOG}" 2>&1 &
echo "PID $!"
