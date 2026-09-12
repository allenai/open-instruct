#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_olmo3_preparation "${1:?Image required}" "${@:2}"
