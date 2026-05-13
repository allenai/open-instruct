#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum_longer_easy_bootstrap}"
export CURRICULUM_BOOTSTRAP_STEPS="${CURRICULUM_BOOTSTRAP_STEPS:-200}"
export CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-200}"

exec bash "${SCRIPT_DIR}/qwen3_4b_dapo_math_dc.sh" "$@"
