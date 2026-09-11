#!/bin/bash
set -euo pipefail
python scripts/miles/launch_hero_conversion.py "${1:?Pass the image from the build wrapper}" "${@:2}"
