#!/bin/bash
# H010 matched tokenizer ablation. Invoke through build_image_and_launch.sh.
# The legacy image is immutable and is the successful H008 anchor's image.
# Core code, lockfile and model config are unchanged between that image's
# 1d8658f87 and the b1714871b parent of this branch. Only aligned uses the fix.
set -euo pipefail
BUILT_IMAGE="${1:?built image required}"
ARM="${2:?aligned or legacy required}"
MODE="${3:-train}"
# H008 anchor-lr5e-5 uses 34521; its seed2 companion uses 34522. SEED (33333) is
# a tokenization cache key and must never move -- vary the data order only.
# Set before the case block: train_full puts the seed in the output dir.
DATA_LOADER_SEED="${DATA_LOADER_SEED:-34521}"
# EXPERIMENT=h015 gives each arm, mode and data seed its own run name, output dir and
# convert source; the h010 defaults put control and seed2 in one existing dir.
EXPERIMENT="${EXPERIMENT:-h010}"
BUDGET="${BUDGET:-}"
H015_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895
# Local, never inherited: an exported RUN_TAG must not rename another mode.
RUN_TAG=""
# The arm alone decides THINK_TOKENS (a tokenization cache key): an inherited value must
# never turn an aligned control into a think run. EXPECTED_NUMPY_CACHE makes a job built
# from this branch die before tokenizing if its arguments resolve to a different cache than
# the arm's; the immutable legacy image predates the check and ignores it.
case "$ARM" in
    aligned) IMAGE="$BUILT_IMAGE"; THINK_TOKENS=0; ARM_CACHE=062b8a3d20-6068a350 ;;
    legacy) IMAGE=01M2KSSB3FCYJ8PNB7B672N9SP; THINK_TOKENS=0; ARM_CACHE=15bfc110a1-6068a350 ;;
    # H015: aligned plus single-token <think>/</think> in reserved slots (#1911). Its cache
    # hash is known only once the tokenize job has run; pass it as THINK_CACHE.
    think)
        IMAGE="$BUILT_IMAGE"; THINK_TOKENS=1; ARM_CACHE="${THINK_CACHE:-}"
        case "$ARM_CACHE" in
            062b8a3d20-6068a350|15bfc110a1-6068a350)
                echo "THINK_CACHE=$ARM_CACHE is a flag-off cache; the think arm needs its own" >&2; exit 1 ;;
        esac
        ;;
    *) echo "Unknown arm: $ARM (expected aligned, legacy or think)" >&2; exit 1 ;;
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
        RUN_TAG="train-full-s${DATA_LOADER_SEED}"
        BUDGET=full
        MODE=train
        ;;
    tokenize)
        # CPU-only; builds the arm's own numpy cache (THINK_TOKENS changes its key).
        MODE=tokenize_full
        ;;
    gate)
        export STEPS=30 NNODES=1 NPROC=8 CKPT_STEPS=1000000 EPHEMERAL_STEPS=-1
        export JOB_TIMEOUT=45m
        ;;
    convert)
        # Both arms use the current converter; only legacy training uses H008's image.
        IMAGE="$BUILT_IMAGE"
        export JOB_TIMEOUT=2h CONVERT_GPUS=1 CONVERT_DEVICE=cuda CONVERT_CLUSTER=ai2/holmes
        export CONVERT_PYTHONPATH=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-anchor/olmo-core-b1fd2c97/src
        if [[ "$EXPERIMENT" == "h015" ]]; then
            # BUDGET=full converts the train_full dir of the same arm and seed.
            export CKPT_ROOT="${CKPT_ROOT:-$H015_ROOT/h015-${ARM}-train${BUDGET:+-$BUDGET}-s${DATA_LOADER_SEED}}"
        else
            export CKPT_ROOT="${CKPT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-train-20260918}"
        fi
        export STEP="${STEP:-step3072}"
        if [[ "$ARM" == "legacy" ]]; then
            CACHE=15bfc110a1-6068a350
        elif [[ "$ARM" == "think" ]]; then
            # The tokenizer saved in the think cache carries the promoted slots, and the
            # export must ship that one.
            CACHE="${THINK_CACHE:?set THINK_CACHE to the think arm numpy cache dir name}"
        else
            CACHE=062b8a3d20-6068a350
        fi
        export CONVERT_TOKENIZER="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft/$CACHE/tokenizer"
        ;;
    *) echo "Expected train, train_full, tokenize, gate or convert" >&2; exit 1 ;;
esac
export BASE=hero-small-nonemo SEQ=65536 LR=5e-5
export THINK_TOKENS
# Train and gate must name their cache. Tokenize names it when it is already known, which
# makes a flag-off tokenize job a CPU preflight: the production hash either resolves to the
# arm's cache (and exits "nothing to do") or raises -- before any GPU is reserved.
if [[ -n "$ARM_CACHE" ]]; then
    export EXPECTED_NUMPY_CACHE="$ARM_CACHE"
elif [[ "$MODE" == "train" || "$MODE" == "gate" ]]; then
    echo "set THINK_CACHE to the think arm numpy cache dir name" >&2; exit 1
else
    unset EXPECTED_NUMPY_CACHE
fi
export DATA_LOADER_SEED
export CLUSTER=ai2/holmes WORKSPACE=ai2/olmo-instruct PREEMPTIBLE=0
export PRIORITY="${PRIORITY:-normal}"
# A 2x8 job needs EVERY replica ready within 10 min or Beaker cancels the group
# ("timed out after waiting 10m0s for synchronized replica start") -- which killed
# 01M332MXC7PKM3WWCHCW5WW4QM at 23:07 UTC on 2026-09-21 before step 1, replica 1
# having scheduled but never readied. Retries are the fix for that, but a retry
# RESTARTS the command, and olmo-core only resumes when RESUME_FROM is set -- so a
# mid-run retry silently trains from step 0 again. Default stays 0 for that reason;
# override to 1 when the run has not started yet and a start-time failure is the
# risk being managed, then watch the step counter on the retry.
export MAX_RETRIES="${MAX_RETRIES:-0}"
export KEEP_LAST_N="${KEEP_LAST_N:-1}"
if [[ "$EXPERIMENT" == "h015" ]]; then
    H015_TAG="${ARM}-${MODE}${BUDGET:+-$BUDGET}-s${DATA_LOADER_SEED}"
    export RUN_NAME="${RUN_NAME:-hero-sft-h015-$H015_TAG}"
    export OUTPUT_DIR="${OUTPUT_DIR:-$H015_ROOT/h015-$H015_TAG}"
fi
export RUN_NAME="${RUN_NAME:-hero-sft-h010-${ARM}-${RUN_TAG:-${MODE}-s${DATA_LOADER_SEED}}-20260918}"
export OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-${RUN_TAG:-${MODE}}-20260918}"
# Caller accounts for all queued/running jobs against 32 urgent + 32 normal.
# Explicit interpreter avoids local sync of the separately pinned MoE runtime.
export PY="${PY:-python}"
bash scripts/train/debug/oc_sft_olmoe3_kda_think.sh "$IMAGE" "$MODE"
