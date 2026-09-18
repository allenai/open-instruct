#!/bin/bash
# Add H010's development tasks; the historical dev/paper battery stays separate
# in eval_olmoe3_hero_battery.sh with its original harness revisions and settings.
# Caller reserves capacity before each invocation. One experiment per mode.
set -euo pipefail
CKPT="${1:?HF checkpoint directory required}"
LABEL="${2:?run label required}"
MODE="${3:?omega, lcb, ifeval, gpqa, popqa, or ruler required}"
shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
export CLUSTERS="${CLUSTERS:--c ai2/ceres -c ai2/jupiter -c ai2/saturn}"
export WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
export PRIORITY="${PRIORITY:-normal}"
export GPUS="${GPUS:-4}"
export NUM_INSTANCES="${NUM_INSTANCES:-$GPUS}" TIMEOUT="${TIMEOUT:-6h}"
case "$MODE" in
    omega)
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-omega" \
            -t omega_500:hillclimb -o max_tokens=32768 -o num_samples=1 \
            -t omega_500_out -o max_tokens=32768 -o num_samples=1 \
            --no-preemptible "$@"
        ;;
    lcb)
        export HARNESS=codex_python EVAL_IMAGE=01KVTPSG86RYVXV13FGQDM9GXT PTXAS_PATH=
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-lcb-v3" \
            -o sandboxes.0.inject_swerex=true -o sandboxes.0.instances=4 \
            -o sandboxes.0.startup_timeout=900 -o sandboxes.0.command_timeout=900 \
            -t livecodebench:lite -o max_tokens=32768 -o num_samples=1 \
            --no-preemptible "$@"
        ;;
    ifeval)
        export OLMO_EVAL_DIR="${OLMO_EVAL_DIR_IFEVAL:?Pinned H008 ifeval checkout required}"
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-screen-ifeval" -t ifeval --no-preemptible "$@"
        ;;
    gpqa)
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-screen-gpqa" \
            -t gpqa_main:cot -o max_tokens=32768 -o num_samples=1 --no-preemptible "$@"
        ;;
    popqa)
        export OLMO_EVAL_DIR="${OLMO_EVAL_DIR_PAPER:?Pinned H008 paper checkout required}"
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-screen-popqa" \
            -t popqa:chat -o max_tokens=32768 -o strip_thinking=true -o limit=2000 -o seed=42 --no-preemptible "$@"
        ;;
    ruler)
        export OLMO_EVAL_DIR="${OLMO_EVAL_DIR_DEV:?Pinned H008 dev checkout required}"
        bash "$HERE/eval_olmoe3_hero.sh" "$CKPT" "$LABEL-screen-ruler64k" \
            -t ruler_all__65536 --no-preemptible "$@"
        ;;
    *) echo "Expected omega, lcb, ifeval, gpqa, popqa, or ruler" >&2; exit 1 ;;
esac
