#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python scripts/miles/launch_gsm8k_test_eval.py "$@"
