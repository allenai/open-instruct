#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum_adaptive_strong}"
export CURRICULUM_ADAPTIVE="${CURRICULUM_ADAPTIVE:-true}"
# Pull harder toward live curriculum stats than the light variant.
export CURRICULUM_ADAPTIVE_UPDATE_EVERY="${CURRICULUM_ADAPTIVE_UPDATE_EVERY:-10}"
export CURRICULUM_ADAPTIVE_LEARNING_WEIGHT="${CURRICULUM_ADAPTIVE_LEARNING_WEIGHT:-0.9}"
export CURRICULUM_ADAPTIVE_EXPLORATION_WEIGHT="${CURRICULUM_ADAPTIVE_EXPLORATION_WEIGHT:-0.1}"
export CURRICULUM_ADAPTIVE_BLEND="${CURRICULUM_ADAPTIVE_BLEND:-0.75}"

exec bash "${SCRIPT_DIR}/qwen3_4b_dapo_math_dc.sh" "$@"
