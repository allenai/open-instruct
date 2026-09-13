#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_readiness_cpu "${1:?Pass the image from the build wrapper}" "${@:2}"
