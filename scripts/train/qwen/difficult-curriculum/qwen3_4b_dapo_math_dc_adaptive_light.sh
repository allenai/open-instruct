#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum_adaptive_light}"
export CURRICULUM_ADAPTIVE="${CURRICULUM_ADAPTIVE:-true}"
export CURRICULUM_ADAPTIVE_UPDATE_EVERY="${CURRICULUM_ADAPTIVE_UPDATE_EVERY:-20}"
export CURRICULUM_ADAPTIVE_BLEND="${CURRICULUM_ADAPTIVE_BLEND:-0.25}"

exec bash "${SCRIPT_DIR}/qwen3_4b_dapo_math_dc.sh" "$@"
