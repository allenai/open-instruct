#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python scripts/miles/launch_diagnostic_retention.py "$@"
