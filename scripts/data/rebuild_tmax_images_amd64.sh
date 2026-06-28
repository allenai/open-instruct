#!/bin/bash
set -euo pipefail

# Rebuild and republish tmax task images for x86_64 nodes.
# Usage:
#   bash scripts/data/rebuild_tmax_images_amd64.sh [extra build_tmax_images.py args...]

uv run python scripts/data/build_tmax_images.py \
  --input hamishivi/swerl-tmax-15k-verified \
  --output-dataset hamishivi/swerl-tmax-15k-verified \
  --registry hamishi740 \
  --repo-prefix swerl-tmax \
  --platform linux/amd64 \
  "$@"
