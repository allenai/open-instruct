#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXP="${EXP:-phase2_all_pass}"

exec "${SCRIPT_DIR}/qwen3_4b_phase1_has_8gpu.sh" \
    --model_name_or_path "" \
    --manufactoria_scoring_mode all_pass \
    "$@"
