#!/bin/bash
set -euo pipefail
python scripts/miles/launch_frozen_core_score_profile.py "${1:?Pass the immutable image from the build wrapper}" "${@:2}"
