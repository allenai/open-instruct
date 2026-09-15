#!/bin/bash

# Strictly on-policy ("synchronous") reruns of the two canonical Qwen3.5 math
# pure-OPD runs, changing only the rollout schedule:
#
#   ARM=2b  verifier-DPPO 2B teacher -> Qwen3.5-2B student, LR 1e-6
#           (canonical: W&B v171addf, Beaker 01M1WQ2DRFJF2C1019HMZ02318)
#   ARM=4b  verifier-DPPO 9B teacher -> Qwen3.5-4B student, LR 5e-7
#           (canonical: W&B rdcupvki, Beaker 01M22AHC3VMFDQVBFJCQHPPZ6G)
#
# The canonical runs used async_steps=4 with in-flight weight updates, so each
# batch was sampled by weights up to four updates old and single responses could
# span a weight swap. These reruns set async_steps=1, disable in-flight updates
# and enable --synchronous_rollouts, so every batch is sampled from exactly the
# policy that step trains. Checkpoints are saved every 10 steps (instead of 20)
# for the matched greedy post-hoc evaluation. Everything else (image, DAPO
# split, teacher, seed, prompts, lengths, loss, evals) is pinned to the
# canonical campaign.
#
# --synchronous_rollouts lives in open_instruct/{data_loader,grpo_fast,actor_manager}.py
# on this branch, which is newer than the pinned image, so a code-patch Beaker
# dataset built from this checkout is required:
#   scripts/general_agent/terminal/rl/make_math_patch_dataset.sh
#
# Usage:
#   PATCH_DATASET=<patch-dataset-id> ARM=2b [RUN_MODE=smoke|full] \
#     scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh [extra grpo_fast args]
set -euo pipefail

ARM="${ARM:?Set ARM=2b or ARM=4b}"
PATCH_DATASET="${PATCH_DATASET:?Set PATCH_DATASET to the code-patch Beaker dataset built from this checkout}"
BEAKER_IMAGE="${BEAKER_IMAGE:-01M01EPKXMXR4502S1HNJVYN0M}"
DAPO_SPLIT_DATASET="${DAPO_SPLIT_DATASET:-01M1TKR7BQE4D5CYKYM1TZX0AM}"
export WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
export CLUSTER="${CLUSTER:-ai2/jupiter}"
export PRIORITY="${PRIORITY:-urgent}"
export PATCH_DATASET

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SYNC_ARGS=(
    --async_steps 1
    --inflight_updates false
    --synchronous_rollouts true
    --save_freq 10
)

case "$ARM" in
    2b)
        export OBJECTIVE=opd
        export RUN_MODE="${RUN_MODE:-full}"
        export TEACHER_MODEL="${TEACHER_MODEL:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/qwen35_2b_verifier_dppo_math_fixed_dapo_100step_4node__42__1788675225}"
        if [[ "$RUN_MODE" == "smoke" ]]; then
            export EXP_NAME="${EXP_NAME:-qwen35_2b_opd_from_verifier_2b_sync_smoke_1node}"
            # The smoke recipe already uses async_steps=1 and saves nothing.
            SYNC_ARGS=(--inflight_updates false --synchronous_rollouts true)
        else
            export EXP_NAME="${EXP_NAME:-qwen35_2b_opd_from_verifier_2b_sync_lr1e6_100step_4node}"
        fi
        exec "$HERE/qwen35_2b_math_objective_control_4node.sh" \
            "$BEAKER_IMAGE" "$DAPO_SPLIT_DATASET" "${SYNC_ARGS[@]}" "$@"
        ;;
    4b)
        export EXP_NAME="${EXP_NAME:-qwen35_4b_opd_from_verifier_9b_sync_lr5e7_100step_4node}"
        export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
        exec "$HERE/qwen35_4b_opd_from_verifier_9b_math_4node.sh" \
            "$BEAKER_IMAGE" "$DAPO_SPLIT_DATASET" "${SYNC_ARGS[@]}" "$@"
        ;;
    *)
        echo "ARM must be '2b' or '4b', got '$ARM'" >&2
        exit 2
        ;;
esac
