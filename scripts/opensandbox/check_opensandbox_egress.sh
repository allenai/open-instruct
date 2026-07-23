#!/bin/bash
# Check whether this host can reach the OpenSandbox service and run a sandbox
# end-to-end.
#
# This is the feasibility gate for using OpenSandboxBackend in RL training
# (see docs/sandbox_management.md): the training job must have outbound
# network access to the self-hosted OpenSandbox endpoint (e.g. on GKE
# Autopilot) and be able to drive the full sandbox lifecycle
# (auth -> sandbox create -> exec -> kill).
#
# Usage (e.g. inside a Beaker session/job on the target cluster):
#   export SWERL_OPENSANDBOX_DOMAIN=sandbox.example.com   # or https://...
#   export SWERL_OPENSANDBOX_PROTOCOL=https               # optional, default http
#   export OPEN_SANDBOX_API_KEY=...                       # from Beaker secret
#   bash scripts/opensandbox/check_opensandbox_egress.sh
#
# Requires python3. Installs the opensandbox package into a throwaway venv if
# it isn't already importable (which also exercises PyPI egress).

set -uo pipefail

echo "=== OpenSandbox egress check ($(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

if [ -z "${SWERL_OPENSANDBOX_DOMAIN:-}" ]; then
    echo "RESULT: FAIL — SWERL_OPENSANDBOX_DOMAIN is not set."
    exit 1
fi

# --- Step 1: raw TCP (+TLS for https) reachability (stdlib only, no key) ----
python3 - <<'EOF'
import os
import socket
import ssl
import sys
import time
from urllib.parse import urlparse

domain = os.environ["SWERL_OPENSANDBOX_DOMAIN"]
protocol = os.environ.get("SWERL_OPENSANDBOX_PROTOCOL", "http")
if "://" in domain:
    parsed = urlparse(domain)
    protocol = parsed.scheme
    host, port = parsed.hostname, parsed.port
else:
    parsed = urlparse(f"{protocol}://{domain}")
    host, port = parsed.hostname, parsed.port
if port is None:
    port = 443 if protocol == "https" else 80

start = time.perf_counter()
try:
    with socket.create_connection((host, port), timeout=10) as sock:
        if protocol == "https":
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                pass
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[PASS] reached {host}:{port} over {protocol} ({elapsed_ms:.0f}ms)")
except Exception as e:
    print(f"[FAIL] cannot reach {host}:{port} — {type(e).__name__}: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then
    echo "RESULT: FAIL — no network path to the OpenSandbox service. OpenSandboxBackend cannot work from this host."
    exit 1
fi

# --- Step 2: make sure the opensandbox package is importable -----------------
if [ -z "${OPEN_SANDBOX_API_KEY:-}" ]; then
    echo "RESULT: PARTIAL — network reachability OK, but OPEN_SANDBOX_API_KEY is"
    echo "not set, so the sandbox lifecycle was not tested. Export it and rerun."
    exit 2
fi

PYTHON=python3
if ! $PYTHON -c "import opensandbox" 2>/dev/null; then
    VENV_DIR="$(mktemp -d)/opensandbox-egress-venv"
    echo "opensandbox not importable; installing into throwaway venv $VENV_DIR (tests PyPI egress)..."
    python3 -m venv "$VENV_DIR" && "$VENV_DIR/bin/pip" install --quiet opensandbox
    if [ $? -ne 0 ]; then
        echo "RESULT: FAIL — could not pip install opensandbox (PyPI egress problem?)."
        exit 1
    fi
    PYTHON="$VENV_DIR/bin/python"
fi
echo "[PASS] opensandbox package available"

# --- Step 3: full sandbox lifecycle ------------------------------------------
$PYTHON - <<'EOF'
import os
import sys
import time
from datetime import timedelta

from opensandbox import SandboxSync
from opensandbox.config import ConnectionConfigSync


def timed(name, fn):
    start = time.perf_counter()
    result = fn()
    print(f"[PASS] {name} ({time.perf_counter() - start:.2f}s)")
    return result


config = ConnectionConfigSync(
    domain=os.environ["SWERL_OPENSANDBOX_DOMAIN"],
    protocol=os.environ.get("SWERL_OPENSANDBOX_PROTOCOL", "http"),
    request_timeout=timedelta(seconds=60),
)

try:
    sandbox = timed(
        "SandboxSync.create (python:3.12-slim, 0.5 cpu / 1 GiB)",
        lambda: SandboxSync.create(
            "python:3.12-slim",
            timeout=timedelta(seconds=300),
            ready_timeout=timedelta(seconds=180),
            resource={"cpu": "0.5", "memory": "1024Mi"},
            metadata={"open_instruct": "egress_check"},
            connection_config=config,
        ),
    )
    try:
        # A couple of execs to get a realistic per-command latency sample.
        for i in range(3):
            execution = timed(f"exec #{i + 1} (echo)", lambda: sandbox.commands.run("echo egress-ok"))
            output = "".join(message.text for message in execution.logs.stdout)
            if execution.exit_code != 0 or "egress-ok" not in output:
                print(f"[FAIL] exec returned unexpectedly: exit={execution.exit_code} stdout={output!r}")
                sys.exit(1)
    finally:
        timed("SandboxSync.kill", sandbox.kill)
        sandbox.close()
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    print("RESULT: FAIL — the OpenSandbox service is reachable but the sandbox lifecycle failed.")
    sys.exit(1)

print("RESULT: PASS — this host can run OpenSandboxBackend sandboxes end-to-end.")
EOF
exit $?
