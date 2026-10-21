#!/bin/bash
# H099 (14T) / H100 (4T) LC-replay arms: H045's olmo35 train_full launch (H075's for the 14T base)
# plus one extra numpy dir of long-document replay rows. Invoke through build_image_and_launch.sh,
# which builds the image from this branch (the --extra_numpy_dirs code is not in H045's image):
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 scripts/train/debug/h099_replay_launch.sh 14t 34521
#
# Only these differ from the H045/H075 launches: MODEL (per base), RUN_NAME, OUTPUT_DIR,
# EXTRA_NUMPY_DIRS, TRAIN_FULL_STEPS (S' = 13,409; warmup_ratio and the linear decay are
# fractions of max_train_steps, so the schedule keeps its shape), PRIORITY=high and JOB_TIMEOUT.
# Seeds, cache, LR, warmup ratio, batch, seq, image CUDA version, 2x8 holmes and PREEMPTIBLE=0
# (mason's only positive minRuntime, 8 h) come unchanged from oc_sft_hero_hillclimb_1895.sh.
set -euo pipefail
BUILT_IMAGE="${1:?built image required}"
BASE_TAG="${2:?14t or 4t required}"
SEED="${3:?data loader seed (34521 or 34522) required}"
case "$SEED" in 34521|34522) ;; *) echo "seed must be 34521 or 34522" >&2; exit 1 ;; esac
case "$BASE_TAG" in
    14t)
        HYP=h099
        export MODEL=/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core
        OUT_ROOT=/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-14t600b/runs
        # H075 ran 7.41 h and 8.17 h on this base under a 9 h timeout; x1.14 is ~9.3 h.
        export JOB_TIMEOUT=12h
        ;;
    4t)
        HYP=h100
        export MODEL=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/emo/step5961
        OUT_ROOT=/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-4t/runs
        # H045 ran 5.89 h and 5.93 h; x1.14 is ~6.8 h.
        export JOB_TIMEOUT=10h
        ;;
    *) echo "Unknown base: $BASE_TAG (expected 14t or 4t)" >&2; exit 1 ;;
esac
export H038_CACHE=1edd51f3e3-6068a350
export TRAIN_FULL_STEPS=13409
export DATA_LOADER_SEED="$SEED"
export MAX_RETRIES=1
export EXPERIMENT="$HYP"
export PRIORITY=high
export RUN_NAME="${HYP}-replay-${BASE_TAG}-s${SEED}"
export OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
export EXTRA_NUMPY_DIRS=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/h099-replay/v1
# An inherited RESUME_FROM must never resume another run; pass H099_RESUME_FROM=<dir>/stepN to
# recover this run after a preemption.
unset RESUME_FROM
if [[ -n "${H099_RESUME_FROM:-}" ]]; then export RESUME_FROM="$H099_RESUME_FROM"; fi
# build_image_and_launch.sh has just run `uv sync` in this checkout.
export PY="${PY:-$PWD/.venv/bin/python}"
exec bash scripts/train/debug/oc_sft_hero_hillclimb_1895.sh "$BUILT_IMAGE" olmo35 train_full
