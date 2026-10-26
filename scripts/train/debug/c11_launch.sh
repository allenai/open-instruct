#!/bin/bash
# C11 (ledger H111 14T / H112 4T): H049 nemo-docb's olmo35 launch (ROW_ALIGNED_PARTS=1, DOCB=1) on the C11 mix,
# optionally with the LC-replay extra numpy dir. Invoke through build_image_and_launch.sh:
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 scripts/train/debug/c11_launch.sh <14t|4t> <seed> <c11|c11-replay|c11-decon> [train_full|tokenize]
#
# c11-decon (ledger H119, 14t only, seeds 34521-34524): c11 with the treatment Dolci parquet, its weights, cache and
# steps (C11_DECON_* in c11_config.sh). Its cache is built off the full oe-adapt bucket, so the job reads it from
# OI_NUMPY_CACHE_ROOT.
#
# Everything the registration fixes after the data is built (parquet paths, weights, cache id, steps S and S',
# replay dir) lives in scripts/train/debug/c11_config.sh, committed before any launch; this script refuses to run
# with an unfilled value. Steps follow H099's rule, so warmup_ratio and the linear decay keep their shape.
# PRIORITY=high. C11_MIN_RUNTIME (default 1h) is mason's --min_runtime: above 0 is an allocated holmes job,
# 0 is backfill beyond the allocation (never --preemptible on holmes). A preempted run resumes only by a manual
# relaunch with C11_RESUME_FROM=<dir>/stepNNNN (no --auto_resume: a requeued command would restart at step 0).
set -euo pipefail
BUILT_IMAGE="${1:?built image required}"
BASE_TAG="${2:?14t or 4t required}"
SEED="${3:?data loader seed (34521 or 34522) required}"
ARM_TAG="${4:?c11, c11-replay or c11-decon required}"
MODE="${5:-train_full}"
case "$SEED" in
    34521|34522) ;;
    34523|34524) [[ "$ARM_TAG" == c11-decon ]] || { echo "seeds 34523/34524 are c11-decon only" >&2; exit 1; } ;;
    *) echo "seed must be 34521-34524" >&2; exit 1 ;;
esac
source "$(dirname "$0")/c11_config.sh"
if [[ "$ARM_TAG" == c11-decon ]]; then
    [[ "$BASE_TAG" == 14t ]] || { echo "c11-decon is 14t only" >&2; exit 1; }
    for v in C11_DECON_DOLCI C11_DECON_WN C11_DECON_WS C11_DECON_CACHE C11_DECON_STEPS C11_DECON_CACHE_ROOT; do
        if [[ -z "${!v:-}" || "${!v}" == TODO* ]]; then
            if [[ "$MODE" == tokenize && "$v" =~ ^C11_DECON_(CACHE|STEPS)$ ]]; then continue; fi
            echo "$v is not filled in c11_config.sh" >&2; exit 1
        fi
    done
    C11_DOLCI=$C11_DECON_DOLCI C11_WN=$C11_DECON_WN C11_WS=$C11_DECON_WS
    C11_CACHE=$C11_DECON_CACHE C11_STEPS=$C11_DECON_STEPS
fi
for v in C11_DOLCI C11_NEMO C11_SCI C11_WN C11_WS C11_CACHE C11_STEPS C11_REPLAY_STEPS C11_REPLAY_DIR; do
    if [[ -z "${!v:-}" || "${!v}" == TODO* ]]; then
        if [[ "$MODE" == tokenize && "$v" =~ ^C11_(CACHE|STEPS|REPLAY_STEPS|REPLAY_DIR)$ ]]; then continue; fi
        echo "$v is not filled in c11_config.sh" >&2; exit 1
    fi
done
# Mixer weights must parse as floats (open-instruct reads an integer as a row count).
for v in C11_WN C11_WS; do [[ "${!v}" == *.* ]] || { echo "$v=${!v} must contain a decimal point" >&2; exit 1; }; done
export C11_DOLCI C11_NEMO C11_SCI C11_WN C11_WS
case "$BASE_TAG" in
    14t) export MODEL=/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core
         OUT_ROOT=/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-14t600b/runs; HOURS=11 ;;
    4t)  export MODEL=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/emo/step5961
         OUT_ROOT=/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-4t/runs; HOURS=11 ;;
    *) echo "Unknown base: $BASE_TAG" >&2; exit 1 ;;
esac
case "$ARM_TAG" in
    c11|c11-decon) export TRAIN_FULL_STEPS="${C11_STEPS:-}"; unset EXTRA_NUMPY_DIRS ;;
    c11-replay) export TRAIN_FULL_STEPS="${C11_REPLAY_STEPS:-}"; export EXTRA_NUMPY_DIRS="$C11_REPLAY_DIR"; HOURS=$((HOURS + 2)) ;;
    *) echo "Unknown arm: $ARM_TAG" >&2; exit 1 ;;
esac
export ROW_ALIGNED_PARTS=1 DOCB=1
# oe-adapt-default is full (10-23): mason's HF_* caches there fail with ENOSPC while datasets reads the parquets.
HF_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache
export MASON_EXTRA_ENV="--env HF_HOME=$HF_ROOT/huggingface --env HF_DATASETS_CACHE=$HF_ROOT/huggingface --env HF_HUB_CACHE=$HF_ROOT/hub"
if [[ "$ARM_TAG" == c11-decon ]]; then
    export OI_NUMPY_CACHE_ROOT="$C11_DECON_CACHE_ROOT"
    MASON_EXTRA_ENV="$MASON_EXTRA_ENV --env OI_NUMPY_CACHE_ROOT=$C11_DECON_CACHE_ROOT"
fi
export H038_CACHE="${C11_CACHE:-}"
export DATA_LOADER_SEED="$SEED" MAX_RETRIES=1 PRIORITY=high PREEMPTIBLE=0 JOB_TIMEOUT="${HOURS}h"
export MIN_RUNTIME="${C11_MIN_RUNTIME:-1h}"; unset AUTO_RESUME
export EXPERIMENT="c11"
export RUN_NAME="${ARM_TAG}-${BASE_TAG}-s${SEED}"
export OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
unset RESUME_FROM
if [[ -n "${C11_RESUME_FROM:-}" ]]; then export RESUME_FROM="$C11_RESUME_FROM"; fi
export PY="${PY:-$PWD/.venv/bin/python}"
exec bash scripts/train/debug/oc_sft_hero_hillclimb_1895.sh "$BUILT_IMAGE" olmo35-c11 "$MODE"
