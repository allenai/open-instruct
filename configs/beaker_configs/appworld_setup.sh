#!/bin/bash

set -euo pipefail

export PYTHONPATH="${REPO_PATH:-${PYTHONPATH:-}}"
export PATH="/root/.local/bin:$PATH"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/root/.appworld}"

echo "[appworld_setup] APPWORLD_ROOT=${APPWORLD_ROOT}"
mkdir -p "${APPWORLD_ROOT}"

echo "[appworld_setup] Installing AppWorld extra dependencies..."
uv sync --extra appworld

echo "[appworld_setup] Verifying AppWorld CLI via uv..."
uv run --extra appworld appworld --help >/dev/null

SETUP_MARKER="${APPWORLD_ROOT}/.open_instruct_appworld_ready"
if [[ -f "${SETUP_MARKER}" ]]; then
    echo "[appworld_setup] Existing setup marker found; skipping install/data download."
else
    echo "[appworld_setup] Running \`appworld install\` via uv run..."
    uv run --extra appworld appworld install
    echo "[appworld_setup] Running \`appworld download data\` via uv run..."
    uv run --extra appworld appworld download data
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SETUP_MARKER}"
    echo "[appworld_setup] Setup complete."
fi
