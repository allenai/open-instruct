#!/bin/bash
set -euo pipefail

# Rebuild and republish non-verified tmax task images as multi-platform tags.
# Usage:
#   bash scripts/data/rebuild_tmax_images_multiarch_10k.sh [extra build_tmax_images.py args...]

uv run python scripts/data/build_tmax_images.py \
  --input osieosie/tmax-tasks-skill-taxonomy-20260401-10k \
  --output-dataset hamishivi/swerl-tmax-15k \
  --registry hamishi740 \
  --repo-prefix swerl-tmax \
  --platform linux/amd64,linux/arm64 \
  --use-buildx \
  "$@"
