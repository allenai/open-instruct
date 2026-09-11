#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_control_reaudit "${1:?Image}" "${2:?Retained experiment ID}" "${@:3}"
