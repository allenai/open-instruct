#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_judge_preparation "${1:?Image required}" "${@:2}"
