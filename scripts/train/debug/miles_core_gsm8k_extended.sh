#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_extended_gsm8k "${1:?Pass the image from the build wrapper}" "${@:2}"
