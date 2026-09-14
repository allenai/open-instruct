#!/usr/bin/env bash
set -euo pipefail
exec python -m scripts.miles.launch_inference_capacity "$@"
