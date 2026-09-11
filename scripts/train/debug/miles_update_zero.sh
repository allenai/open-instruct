#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python scripts/miles/launch_update_zero.py "$@"
