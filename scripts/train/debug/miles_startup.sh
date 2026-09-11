#!/usr/bin/env bash
set -euo pipefail
python -m scripts.miles.launch_startup_trial "${1:?Pass the built image}" "${@:2}"
