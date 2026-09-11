#!/usr/bin/env bash
set -euo pipefail
python -m scripts.miles.launch_core_cache_trial "${1:?Pass the image from the build wrapper}" "${@:2}"
