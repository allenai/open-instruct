#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_packing "${1:?Pass the image from the build wrapper}"
