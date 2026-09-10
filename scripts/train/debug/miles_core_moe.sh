#!/bin/bash
set -euo pipefail
# Called by build_image_and_launch.sh --miles after committing and baking sources.
# Use the CLI schema directly: the installed mason SDK cannot encode minRuntime.
python scripts/miles/launch_trial.py "${1:?Pass the image from the build wrapper}"
