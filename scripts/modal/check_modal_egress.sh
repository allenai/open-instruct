#!/bin/bash
# Check whether this host can reach Modal and run a sandbox end-to-end.
#
# This is the feasibility gate for using ModalBackend in RL training
# (see docs/sandbox_management.md): the training job must have outbound
# network access to api.modal.com and be able to drive the full sandbox
# lifecycle (auth -> app lookup -> sandbox create -> exec -> terminate).
#
# Usage (e.g. inside a Beaker session/job on the target cluster):
#   export MODAL_TOKEN_ID=ak-...        # from ~/.modal.toml or Beaker secret
#   export MODAL_TOKEN_SECRET=as-...
#   bash scripts/modal/check_modal_egress.sh
#
# Requires python3. Installs the modal package into a throwaway venv if it
# isn't already importable (which also exercises PyPI egress).

set -uo pipefail

echo "=== Modal egress check ($(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

# --- Step 1: raw TCP/TLS reachability (stdlib only, no credentials) ---------
python3 - <<'EOF'
import socket
import ssl
import sys
import time

failures = 0
for host in ("api.modal.com", "modal.com"):
    start = time.perf_counter()
    try:
        with socket.create_connection((host, 443), timeout=10) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"[PASS] TLS handshake with {host}:443 ({elapsed_ms:.0f}ms)")
    except Exception as e:
        failures += 1
        print(f"[FAIL] cannot reach {host}:443 — {type(e).__name__}: {e}")
sys.exit(1 if failures else 0)
EOF
if [ $? -ne 0 ]; then
    echo "RESULT: FAIL — no network path to Modal. ModalBackend cannot work from this host."
    exit 1
fi

# --- Step 2: make sure the modal package is importable ----------------------
if [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; then
    echo "RESULT: PARTIAL — TLS reachability OK, but MODAL_TOKEN_ID/MODAL_TOKEN_SECRET"
    echo "are not set, so the sandbox lifecycle was not tested. Export them and rerun."
    exit 2
fi

PYTHON=python3
if ! $PYTHON -c "import modal" 2>/dev/null; then
    VENV_DIR="$(mktemp -d)/modal-egress-venv"
    echo "modal not importable; installing into throwaway venv $VENV_DIR (tests PyPI egress)..."
    python3 -m venv "$VENV_DIR" && "$VENV_DIR/bin/pip" install --quiet modal
    if [ $? -ne 0 ]; then
        echo "RESULT: FAIL — could not pip install modal (PyPI egress problem?)."
        exit 1
    fi
    PYTHON="$VENV_DIR/bin/python"
fi
echo "[PASS] modal package available ($($PYTHON -c 'import modal; print(modal.__version__)'))"

# --- Step 3: full sandbox lifecycle against the agent-training environment --
$PYTHON - <<'EOF'
import sys
import time

import modal

# Reuse the app ModalBackend itself uses so this check doesn't leave an
# extra deployed app on the Modal dashboard.
APP_NAME = "open-instruct-sandbox"
ENVIRONMENT = "agent-training"


def timed(name, fn):
    start = time.perf_counter()
    result = fn()
    print(f"[PASS] {name} ({time.perf_counter() - start:.2f}s)")
    return result


try:
    app = timed(
        f"App.lookup (auth + env={ENVIRONMENT!r})",
        lambda: modal.App.lookup(APP_NAME, create_if_missing=True, environment_name=ENVIRONMENT),
    )
    sandbox = timed(
        "Sandbox.create (python:3.12-slim, 0.5 core / 1 GiB)",
        lambda: modal.Sandbox.create(
            app=app,
            image=modal.Image.from_registry("python:3.12-slim"),
            timeout=300,
            cpu=0.5,
            memory=1024,
        ),
    )
    try:
        # A couple of execs to get a realistic per-command latency sample.
        for i in range(3):
            process = timed(f"exec #{i + 1} (echo)", lambda: sandbox.exec("echo", "egress-ok", text=False))
            output = process.stdout.read()
            if process.wait() != 0 or b"egress-ok" not in output:
                print(f"[FAIL] exec returned unexpectedly: exit={process.returncode} stdout={output!r}")
                sys.exit(1)
    finally:
        timed("Sandbox.terminate", sandbox.terminate)
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    print("RESULT: FAIL — Modal API reachable checks passed but the sandbox lifecycle failed.")
    sys.exit(1)

print("RESULT: PASS — this host can run ModalBackend sandboxes end-to-end.")
EOF
exit $?
