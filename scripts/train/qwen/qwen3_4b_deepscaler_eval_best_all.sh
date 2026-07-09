#!/bin/bash

# Launch eval-only jobs (BRUMO/HMMT/AIME 2025) for all 20 deepscaler seeds at each
# seed's best in-training AIME step (see experiment.md), using HF checkpoints
# converted from the OLMo-core checkpoint states via
# scripts/train/convert_olmo_core_to_hf.py.
#
# Usage: ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_eval_best_all.sh

set -euo pipefail

BEAKER_IMAGE="${1:?usage: $0 BEAKER_IMAGE}"

HF_BASE="/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/best_aime_hf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# label best_step
RUNS=(
    "baseline_n16_k8_seed1 1100"
    "baseline_n16_k8_seed2 1100"
    "baseline_n16_k8_seed3 1900"
    "baseline_n8_k16_seed1 900"
    "baseline_n8_k16_seed2 1400"
    "baseline_n8_k16_seed3 1500"
    "baseline_n4_k32_seed1 1200"
    "baseline_n4_k32_seed2 800"
    "baseline_n4_k32_seed3 1200"
    "baseline_n2_k64_seed1 1000"
    "baseline_n2_k64_seed2 1600"
    "baseline_n2_k64_seed3 800"
    "ngu05_n8_k16_seed1 1600"
    "ngu05_n8_k16_seed2 1400"
    "ngu05_n8_k16_seed3 1000"
    "ngu075_n8_k16_seed1 1000"
    "ngu075_n8_k16_seed3 1100"
    "ngu0875_n8_k16_seed1 1700"
    "ngu0875_n8_k16_seed2 1700"
    "ngu0875_n8_k16_seed3 1000"
)

for entry in "${RUNS[@]}"; do
    read -r label step <<< "${entry}"
    model_dir="${HF_BASE}/qwen3_4b_base_deepscaler_oc_2k_${label}_step${step}"
    if [ ! -f "${model_dir}/config.json" ]; then
        echo "ERROR: missing converted checkpoint ${model_dir}" >&2
        exit 1
    fi
    echo "Launching eval for ${label} at step ${step}"
    BEAKER_IMAGE="${BEAKER_IMAGE}" \
    MODEL_NAME_OR_PATH="${model_dir}" \
    EXP_NAME="eval_best_${label}" \
    BEST_STEP="${step}" \
    WANDB_GROUP_NAME="deepscaler_eval_best" \
        bash "${SCRIPT_DIR}/qwen3_4b_deepscaler_eval_best.sh"
done
