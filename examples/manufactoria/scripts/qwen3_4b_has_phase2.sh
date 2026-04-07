#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BASE_MODEL="${BASE_MODEL:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/Qwen3_4B_Instruct_manufactoria_has_phase1_pass_rate__1__1775248654_checkpoints/step_200}"
SCORE_MODE=all_pass

export EXP_NAME="${EXP_NAME:-qwen3_4b_it_manufac_phase2_${SCORE_MODE}}"
export RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/qwen3_4b_phase1_has_8gpu.sh" \
    --model_name_or_path "${BASE_MODEL}" \
    --manufactoria_scoring_mode "${SCORE_MODE}" \
    "$@"
