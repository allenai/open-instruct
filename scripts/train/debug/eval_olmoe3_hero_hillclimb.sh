#!/bin/bash
# Add H010's development tasks; the historical dev/paper battery stays separate
# in eval_olmoe3_hero_battery.sh with its original harness revisions and settings.
# Caller reserves capacity before each invocation. One experiment per mode.
set -euo pipefail
CKPT="${1:?HF checkpoint directory required}"
LABEL="${2:?run label required}"
MODE="${3:?omega or lcb required}"
shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
export CLUSTERS='-c ai2/holmes'
export WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}"
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
    *) echo "Expected omega or lcb" >&2; exit 1 ;;
esac
