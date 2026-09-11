#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python scripts/miles/launch_trainer_route_compare.py "$@"
