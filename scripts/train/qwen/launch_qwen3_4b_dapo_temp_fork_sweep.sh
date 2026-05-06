#!/bin/bash
set -euo pipefail

BASE_STEP="${BASE_STEP:-500}"
TEMP_DELTA="${TEMP_DELTA:-0.05}"
NUM_TEMPS_EACH_SIDE="${NUM_TEMPS_EACH_SIDE:-3}"
INCLUDE_BASE_TEMPERATURE="${INCLUDE_BASE_TEMPERATURE:-1}"
SWEEP_CHECKPOINT_PARENT="${SWEEP_CHECKPOINT_PARENT:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/qwen3_4b_base_dapo_temp_fork_sweep_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${BASE_EXPERIMENT_ID:-}" && ( -z "${BASE_CHECKPOINT_STATE_DIR:-}" || -z "${BASE_TEMPERATURE:-}" ) ]]; then
    echo "Set BASE_EXPERIMENT_ID, or set both BASE_CHECKPOINT_STATE_DIR and BASE_TEMPERATURE."
    exit 1
fi

parse_base_spec() {
    python - "$BASE_EXPERIMENT_ID" <<'PY'
import json
import shlex
import subprocess
import sys

experiment_id = sys.argv[1]
raw = subprocess.check_output(["beaker", "experiment", "inspect", experiment_id, "--format", "json"], text=True)
experiment = json.loads(raw)[0]
arguments = experiment["jobs"][0]["execution"]["spec"]["arguments"]
command = " ".join(arguments)
tokens = shlex.split(command)

def get_flag_value(flag: str) -> str | None:
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None

temperature = get_flag_value("--temperature")
checkpoint_state_dir = get_flag_value("--checkpoint_state_dir")
if temperature is None:
    raise SystemExit(f"Could not find --temperature in Beaker experiment {experiment_id}")
if checkpoint_state_dir is None:
    raise SystemExit(f"Could not find --checkpoint_state_dir in Beaker experiment {experiment_id}")
print(temperature)
print(checkpoint_state_dir)
PY
}

if [[ -n "${BASE_EXPERIMENT_ID:-}" ]]; then
    mapfile -t parsed < <(parse_base_spec)
    BASE_TEMPERATURE="${BASE_TEMPERATURE:-${parsed[0]}}"
    BASE_CHECKPOINT_STATE_DIR="${BASE_CHECKPOINT_STATE_DIR:-${parsed[1]}}"
fi

mapfile -t TEMPERATURES < <(
    python - "$BASE_TEMPERATURE" "$TEMP_DELTA" "$NUM_TEMPS_EACH_SIDE" "$INCLUDE_BASE_TEMPERATURE" <<'PY'
import sys

base = float(sys.argv[1])
delta = float(sys.argv[2])
n = int(sys.argv[3])
include_base = sys.argv[4] == "1"
temps = [base - delta * i for i in range(n, 0, -1)]
if include_base:
    temps.append(base)
temps.extend(base + delta * i for i in range(1, n + 1))
for temp in temps:
    if temp <= 0:
        raise SystemExit(f"Temperature must be positive, got {temp}")
    print(f"{temp:.2f}")
PY
)

echo "Base temperature: ${BASE_TEMPERATURE}"
echo "Base checkpoint state dir: ${BASE_CHECKPOINT_STATE_DIR}"
echo "Base checkpoint tag: global_step${BASE_STEP}"
echo "Branch checkpoint parent: ${SWEEP_CHECKPOINT_PARENT}"
echo "Temperatures: ${TEMPERATURES[*]}"

for temperature in "${TEMPERATURES[@]}"; do
    label="$(python - "$temperature" <<'PY'
import sys
print(sys.argv[1].replace(".", "p"))
PY
)"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    export TEMPERATURE="$temperature"
    export RESUME_STEP="$BASE_STEP"
    export BASE_CHECKPOINT_STATE_DIR
    export BRANCH_CHECKPOINT_STATE_DIR="${SWEEP_CHECKPOINT_PARENT}/t${label}"
    export EXP_NAME="qwen3_4b_base_dapo_temp_fork_t${label}"
    export RUN_NAME="${EXP_NAME}_from_step${BASE_STEP}_${timestamp}"
    echo "Launching ${RUN_NAME} with checkpoint ${BRANCH_CHECKPOINT_STATE_DIR}"
    ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_dapo_math_resume_temp_branch.sh
done
