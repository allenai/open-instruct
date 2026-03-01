#!/bin/bash

_appworld_fail() {
    local rc="${1:-1}"
    return "${rc}" 2>/dev/null || exit "${rc}"
}

export PYTHONPATH="${REPO_PATH:-${PYTHONPATH:-}}"
export PATH="/root/.local/bin:$PATH"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/root/.appworld}"
APPWORLD_PACKAGE_REF="${APPWORLD_PACKAGE_REF:-git+https://github.com/stonybrooknlp/appworld.git@main}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "[appworld_setup] APPWORLD_ROOT=${APPWORLD_ROOT}"
mkdir -p "${APPWORLD_ROOT}"

echo "[appworld_setup] Ensuring AppWorld is installed for ${PYTHON_BIN}..."
if ! "${PYTHON_BIN}" -c "import appworld" >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" "${APPWORLD_PACKAGE_REF}" || _appworld_fail $?
fi

echo "[appworld_setup] Verifying AppWorld CLI via uv run..."
uv run --python "${PYTHON_BIN}" appworld --help >/dev/null || _appworld_fail $?

SETUP_MARKER="${APPWORLD_ROOT}/.open_instruct_appworld_ready"
if [[ -f "${SETUP_MARKER}" ]]; then
    echo "[appworld_setup] Existing setup marker found; skipping install/data download."
else
    echo "[appworld_setup] Running \`appworld install\` via uv run..."
    uv run --python "${PYTHON_BIN}" appworld install || _appworld_fail $?
    echo "[appworld_setup] Running \`appworld download data\` via uv run..."
    uv run --python "${PYTHON_BIN}" appworld download data || _appworld_fail $?
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SETUP_MARKER}"
    echo "[appworld_setup] Setup complete."
fi
