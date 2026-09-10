#!/usr/bin/env bash
set -euo pipefail
UV_NO_SYNC=1 uv run python scripts/miles/launch_contract.py "$@"
