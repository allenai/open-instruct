#!/bin/bash
# H010 matched tokenizer ablation. Invoke through build_image_and_launch.sh.
# The legacy image is immutable and is the successful H008 anchor's image.
# Core code, lockfile and model config are unchanged between that image's
# 1d8658f87 and the b1714871b parent of this branch. Only aligned uses the fix.
set -euo pipefail
BUILT_IMAGE="${1:?built image required}"
ARM="${2:?aligned or legacy required}"
MODE="${3:-train}"
case "$ARM" in
    aligned) IMAGE="$BUILT_IMAGE" ;;
    legacy) IMAGE=01M2KSSB3FCYJ8PNB7B672N9SP ;;
    *) echo "Unknown arm: $ARM" >&2; exit 1 ;;
esac
case "$MODE" in
    train)
        export STEPS=3072 NNODES=2 NPROC=8 CKPT_STEPS=3072 EPHEMERAL_STEPS=1024
        export JOB_TIMEOUT=6h
        ;;
    gate)
        export STEPS=30 NPROC=8 CKPT_STEPS=1000000 EPHEMERAL_STEPS=-1
        export JOB_TIMEOUT=45m
        ;;
    convert)
        # Both arms use the current converter; only legacy training uses H008's image.
        IMAGE="$BUILT_IMAGE"
        export JOB_TIMEOUT=2h CONVERT_GPUS=1 CONVERT_DEVICE=cuda CONVERT_CLUSTER=ai2/holmes
        export CONVERT_PYTHONPATH=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-anchor/olmo-core-b1fd2c97/src
        export CKPT_ROOT="${CKPT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-train-20260918}"
        export STEP=step3072
        if [[ "$ARM" == "legacy" ]]; then
            CACHE=15bfc110a1-6068a350
        else
            CACHE=062b8a3d20-6068a350
        fi
        export CONVERT_TOKENIZER="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft/$CACHE/tokenizer"
        ;;
    *) echo "Expected train, gate or convert" >&2; exit 1 ;;
esac
export BASE=hero-small-nonemo SEQ=65536 LR=5e-5 DATA_LOADER_SEED=34521
export CLUSTER=ai2/holmes WORKSPACE=ai2/olmo-instruct PREEMPTIBLE=0
export PRIORITY="${PRIORITY:-normal}" MAX_RETRIES=0 KEEP_LAST_N=1
export RUN_NAME="hero-sft-h010-${ARM}-${MODE}-s34521-20260918"
export OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-${MODE}-20260918}"
# Caller accounts for all queued/running jobs against 32 urgent + 32 normal.
# Explicit interpreter avoids local sync of the separately pinned MoE runtime.
export PY="${PY:-python}"
bash scripts/train/debug/oc_sft_olmoe3_kda_think.sh "$IMAGE" "$MODE"
