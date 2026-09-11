#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_checkpoint_metadata "${1:?Pass the pinned image from the build wrapper}" "${@:2}"
