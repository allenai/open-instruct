#!/bin/bash
# Kill all live OpenSandbox sandboxes tagged with an app name — the janitor to
# run after killing a training job (a killed job cannot kill its own
# sandboxes, which otherwise consume cluster resources until their lifetime
# cap).
#
# Usage:
#   ./scripts/opensandbox/cleanup_opensandbox_sandboxes.sh [app-name]
#
# Defaults to the OpenSandboxBackend default app name (open-instruct-sandbox).
# Requires SWERL_OPENSANDBOX_DOMAIN (and optionally SWERL_OPENSANDBOX_PROTOCOL),
# OPEN_SANDBOX_API_KEY, and the opensandbox package (run via `uv run` in the
# repo, or `pip install opensandbox`).

set -euo pipefail

APP_NAME="${1:-${SWERL_OPENSANDBOX_APP_NAME:-open-instruct-sandbox}}"

python3 - "$APP_NAME" <<'EOF'
import os
import sys
from datetime import timedelta

from opensandbox import SandboxManagerSync
from opensandbox.config import ConnectionConfigSync
from opensandbox.models import SandboxFilter

app_name = sys.argv[1]
domain = os.environ.get("SWERL_OPENSANDBOX_DOMAIN")
if not domain:
    print("SWERL_OPENSANDBOX_DOMAIN is not set.", file=sys.stderr)
    sys.exit(1)

config = ConnectionConfigSync(
    domain=domain,
    protocol=os.environ.get("SWERL_OPENSANDBOX_PROTOCOL", "http"),
    request_timeout=timedelta(seconds=60),
)
manager = SandboxManagerSync.create(config)

count = 0
failures = 0
# Kill page 1 repeatedly: each pass shrinks the live set, so this terminates
# once nothing kills successfully or the listing comes back empty.
while True:
    page = manager.list_sandbox_infos(
        SandboxFilter(
            states=["RUNNING", "PENDING", "PAUSED"],
            metadata={"open_instruct_app": app_name},
            page=1,
            page_size=100,
        )
    )
    if not page.sandbox_infos:
        break
    killed_this_pass = 0
    for info in page.sandbox_infos:
        try:
            manager.kill_sandbox(info.id)
            count += 1
            killed_this_pass += 1
            print(f"killed {info.id}")
        except Exception as e:
            failures += 1
            print(f"FAILED to kill {info.id}: {e}", file=sys.stderr)
    if killed_this_pass == 0:
        break

manager.close()
print(f"{count} sandbox(es) killed for app {app_name!r}; {failures} failure(s)")
sys.exit(1 if failures else 0)
EOF
