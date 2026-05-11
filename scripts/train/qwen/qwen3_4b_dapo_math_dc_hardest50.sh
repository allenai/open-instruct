#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum_hardest50}"
export CURRICULUM_MIN_QUANTILE="${CURRICULUM_MIN_QUANTILE:-0.5}"
export CURRICULUM_MAX_QUANTILE="${CURRICULUM_MAX_QUANTILE:-1.0}"

exec "${SCRIPT_DIR}/qwen3_4b_dapo_math_dc.sh" "$@"
