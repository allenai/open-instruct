#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum_adaptive_strong_hardest50}"
export CURRICULUM_MIN_QUANTILE="${CURRICULUM_MIN_QUANTILE:-0.5}"
export CURRICULUM_MAX_QUANTILE="${CURRICULUM_MAX_QUANTILE:-1.0}"
# After filtering out the easy half, start bootstrap at the easiest remaining
# bucket instead of inheriting the base global target near bucket 0.
export CURRICULUM_BOOTSTRAP_TARGET="${CURRICULUM_BOOTSTRAP_TARGET:-0.5}"

exec bash "${SCRIPT_DIR}/qwen3_4b_dapo_math_dc_adaptive_strong.sh" "$@"
