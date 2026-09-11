#!/usr/bin/env bash
set -euo pipefail
python -m scripts.miles.launch_light_sft_gsm8k "${1:?Pass the image from the build wrapper}" "${@:2}"
