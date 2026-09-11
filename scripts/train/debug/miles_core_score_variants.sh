#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_core_score_variants "${1:?Pass the immutable image from the build wrapper}" "${@:2}"
