#!/bin/bash
# Terminate all live Modal sandboxes for an app — the janitor to run after
# killing a training job (a killed job cannot terminate its own sandboxes,
# which otherwise bill until their lifetime cap).
#
# Usage:
#   ./scripts/modal/cleanup_modal_sandboxes.sh [app-name] [environment]
#
# Defaults to the ModalBackend defaults (open-instruct-sandbox / agent-training).
# Requires Modal credentials (~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET)
# and the modal package (run via `uv run` in the repo, or `pip install modal`).

set -euo pipefail

APP_NAME="${1:-${SWERL_MODAL_APP_NAME:-open-instruct-sandbox}}"
ENVIRONMENT="${2:-${SWERL_MODAL_ENVIRONMENT:-agent-training}}"

python3 - "$APP_NAME" "$ENVIRONMENT" <<'EOF'
import sys

import modal

app_name, environment = sys.argv[1], sys.argv[2]
try:
    app = modal.App.lookup(app_name, environment_name=environment)
except modal.exception.NotFoundError:
    print(f"No app named {app_name!r} in environment {environment!r}; nothing to clean up.")
    sys.exit(0)

count = 0
failures = 0
for sandbox in modal.Sandbox.list(app_id=app.app_id):
    try:
        sandbox.terminate()
        count += 1
        print(f"terminated {sandbox.object_id}")
    except Exception as e:
        failures += 1
        print(f"FAILED to terminate {sandbox.object_id}: {e}", file=sys.stderr)

print(f"{count} sandbox(es) terminated for app {app_name!r} (env {environment!r}); {failures} failure(s)")
sys.exit(1 if failures else 0)
EOF
