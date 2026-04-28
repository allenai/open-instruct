#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/qwen3_4b_dapo_math.sh"

for CLIP_HIGHER in 0.2 0.3 0.4; do
    CLIP_TAG=$(echo "${CLIP_HIGHER}" | tr '.' 'p')
    EXP_NAME="qwen3_4b_base_dapo_clip_higher_${CLIP_TAG}" \
    bash "${BASE_SCRIPT}" \
        --clip_higher "${CLIP_HIGHER}" "$@"
done
