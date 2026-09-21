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
    train_full)
        # The declared aligned-11768 arm: the H008 anchor budget exactly,
        # 11768 x 1,048,576 = 12,339,642,368 tokens. CKPT_STEPS=5884 gives the
        # mid-run (un-annealed) point H008 also has, so both comparisons are
        # schedule-matched. The H008 anchor ran 11768 steps in 6.02 h on 2x8;
        # 9h of timeout leaves room for a slow start without reaching the 8 h
        # minRuntime shield's preemption window unnecessarily early.
        export STEPS=11768 NNODES=2 NPROC=8 CKPT_STEPS=5884 EPHEMERAL_STEPS=1024
        export JOB_TIMEOUT="${JOB_TIMEOUT:-9h}"
        # RUN_TAG keeps the run name and output dir distinct from the 3072-update
        # arm, whose dir already exists; MODE itself must read "train" downstream.
        # Two permanent checkpoints, not one: KEEP_LAST_N=1 would prune step5884
        # the moment step11768 lands, and step5884 is the schedule-matched
        # mid-run comparison against the H008 anchor. 2 x 207 GB; convert and
        # delete each as it lands.
        export KEEP_LAST_N=2
        RUN_TAG=train-full
        MODE=train
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
        export STEP="${STEP:-step3072}"
        if [[ "$ARM" == "legacy" ]]; then
            CACHE=15bfc110a1-6068a350
        else
            CACHE=062b8a3d20-6068a350
        fi
        export CONVERT_TOKENIZER="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft/$CACHE/tokenizer"
        ;;
    *) echo "Expected train, gate or convert" >&2; exit 1 ;;
esac
export BASE=hero-small-nonemo SEQ=65536 LR=5e-5
# H008 anchor-lr5e-5 uses 34521; its seed2 companion uses 34522. SEED (33333) is
# a tokenization cache key and must never move -- vary the data order only.
export DATA_LOADER_SEED="${DATA_LOADER_SEED:-34521}"
export CLUSTER=ai2/holmes WORKSPACE=ai2/olmo-instruct PREEMPTIBLE=0
export PRIORITY="${PRIORITY:-normal}" MAX_RETRIES=0
export KEEP_LAST_N="${KEEP_LAST_N:-1}"
export RUN_NAME="${RUN_NAME:-hero-sft-h010-${ARM}-${RUN_TAG:-${MODE}}-s${DATA_LOADER_SEED}-20260918}"
export OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-${RUN_TAG:-${MODE}}-20260918}"
# Caller accounts for all queued/running jobs against 32 urgent + 32 normal.
# Explicit interpreter avoids local sync of the separately pinned MoE runtime.
export PY="${PY:-python}"
bash scripts/train/debug/oc_sft_olmoe3_kda_think.sh "$IMAGE" "$MODE"
