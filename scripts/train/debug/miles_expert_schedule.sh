#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_expert_schedule "${1:?Pass the image from the committed-image wrapper}"
