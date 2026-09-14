#!/usr/bin/env bash
set -euo pipefail
exec python -m scripts.miles.launch_trainer_capacity "$@"
