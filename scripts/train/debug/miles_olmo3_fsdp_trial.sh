#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_olmo3_fsdp_trial "${1:?Pass the image from the build wrapper}" "${@:2}"
