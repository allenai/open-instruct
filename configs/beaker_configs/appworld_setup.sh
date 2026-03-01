#!/bin/bash

_appworld_fail() {
    local rc="${1:-1}"
    return "${rc}" 2>/dev/null || exit "${rc}"
}

_appworld_cli() {
    uv run --python "${PYTHON_BIN}" python -m appworld.cli --root "${APPWORLD_ROOT}" "$@" || _appworld_fail $?
}

_appworld_has_private_apps() {
    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib

importlib.import_module("appworld.apps.admin.models")
PY
}

_appworld_has_task_data() {
    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os
from pathlib import Path
import sys

root = Path(os.environ["APPWORLD_ROOT"])
tasks_dir = root / "data" / "tasks"
has_tasks = tasks_dir.exists() and any(tasks_dir.iterdir())
sys.exit(0 if has_tasks else 1)
PY
}

export PYTHONPATH="${REPO_PATH:-${PYTHONPATH:-}}"
export PATH="/root/.local/bin:$PATH"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/root/.appworld}"
APPWORLD_PACKAGE_REF="${APPWORLD_PACKAGE_REF:-git+https://github.com/stonybrooknlp/appworld.git@main}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ "${APPWORLD_PACKAGE_REF}" == git+* ]]; then
    export UV_GIT_LFS="${UV_GIT_LFS:-1}"
fi

echo "[appworld_setup] APPWORLD_ROOT=${APPWORLD_ROOT}"
mkdir -p "${APPWORLD_ROOT}"

echo "[appworld_setup] Ensuring AppWorld is installed for ${PYTHON_BIN}..."
if ! "${PYTHON_BIN}" -c "import appworld" >/dev/null 2>&1; then
    UV_GIT_LFS="${UV_GIT_LFS:-1}" uv pip install --python "${PYTHON_BIN}" "${APPWORLD_PACKAGE_REF}" || _appworld_fail $?
fi

echo "[appworld_setup] Verifying AppWorld CLI via uv run..."
_appworld_cli --help

SETUP_MARKER="${APPWORLD_ROOT}/.open_instruct_appworld_ready"
needs_setup=0
if [[ ! -f "${SETUP_MARKER}" ]]; then
    needs_setup=1
fi
if ! _appworld_has_private_apps; then
    echo "[appworld_setup] Missing app modules (e.g., appworld.apps.admin); will run install."
    needs_setup=1
fi
if ! _appworld_has_task_data; then
    echo "[appworld_setup] Missing task data under APPWORLD_ROOT; will run download."
    needs_setup=1
fi

if [[ "${needs_setup}" -eq 0 ]]; then
    echo "[appworld_setup] Existing setup marker found and installation is complete; skipping."
else
    if [[ "${APPWORLD_PACKAGE_REF}" == git+* ]]; then
        echo "[appworld_setup] Reinstalling AppWorld package with UV_GIT_LFS to ensure bundle assets are present..."
        UV_GIT_LFS="${UV_GIT_LFS:-1}" uv pip install --python "${PYTHON_BIN}" --upgrade --force-reinstall "${APPWORLD_PACKAGE_REF}" || _appworld_fail $?
    fi

    echo "[appworld_setup] Running \`appworld install\` via uv run..."
    _appworld_cli install
    echo "[appworld_setup] Running \`appworld download data\` via uv run..."
    _appworld_cli download data

    if ! _appworld_has_private_apps; then
        echo "[appworld_setup] Error: app modules are still missing after install."
        echo "[appworld_setup] This commonly happens when AppWorld is installed from git without git-lfs assets."
        _appworld_fail 1
    fi
    if ! _appworld_has_task_data; then
        echo "[appworld_setup] Error: task data is still missing under APPWORLD_ROOT=${APPWORLD_ROOT}."
        _appworld_fail 1
    fi

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SETUP_MARKER}"
    echo "[appworld_setup] Setup complete."
fi
