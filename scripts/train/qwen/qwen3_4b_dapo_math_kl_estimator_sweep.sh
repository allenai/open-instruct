#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/qwen3_4b_dapo_math.sh"

for KL_ESTIMATOR in 0 1 2 3; do
    EXP_NAME="qwen3_4b_base_dapo_kl_estimator_${KL_ESTIMATOR}" \
    bash "${BASE_SCRIPT}" \
        --kl_estimator "${KL_ESTIMATOR}" "$@"
done
