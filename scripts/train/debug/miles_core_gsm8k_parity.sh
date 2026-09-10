#!/bin/bash
set -euo pipefail
python scripts/miles/launch_gsm8k_parity.py "${1:?Pass the image from the build wrapper}" "${@:2}"
