#!/bin/bash

# Pipelined ("async 4") rerun of the canonical Qwen3.5 2B math pure-OPD run with
# --fixed_prompt_batches: the same async_steps=4 with in-flight weight updates as
# the canonical run (W&B v171addf, Beaker 01M1WQ2DRFJF2C1019HMZ02318), but every
# training step consumes exactly the prompt set queued for it instead of the
# first responses to finish, and nothing is dropped for age. This is the
# verification run for the fix to the async path (OPD validation program step
# 2d): it should track the strictly on-policy rerun
# (scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh) where the
# canonical async run collapsed, at roughly the canonical run's throughput.
#
# Checkpoints are saved every 10 steps for the matched greedy post-hoc
# evaluation. Everything else (image, DAPO split, teacher, seed, prompts,
# lengths, loss, evals) is pinned to the canonical campaign.
#
# --fixed_prompt_batches lives in open_instruct/{data_loader,data_types,vllm_utils,grpo_fast}.py
# on this branch, which is newer than the pinned image, so a code-patch Beaker
# dataset built from this checkout is required (data_types.py and vllm_utils.py
# are now part of the patch):
#   scripts/general_agent/terminal/rl/make_math_patch_dataset.sh
#
# Usage:
#   PATCH_DATASET=<patch-dataset-id> [RUN_MODE=smoke|full] \
#     scripts/general_agent/terminal/rl/qwen35_math_opd_fixed_batches_rerun.sh [extra grpo_fast args]
set -euo pipefail

PATCH_DATASET="${PATCH_DATASET:?Set PATCH_DATASET to the code-patch Beaker dataset built from this checkout}"
BEAKER_IMAGE="${BEAKER_IMAGE:-01M01EPKXMXR4502S1HNJVYN0M}"
DAPO_SPLIT_DATASET="${DAPO_SPLIT_DATASET:-01M1TKR7BQE4D5CYKYM1TZX0AM}"
export WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
export CLUSTER="${CLUSTER:-ai2/jupiter}"
export PRIORITY="${PRIORITY:-urgent}"
export PATCH_DATASET
export OBJECTIVE=opd
export RUN_MODE="${RUN_MODE:-full}"
export TEACHER_MODEL="${TEACHER_MODEL:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/qwen35_2b_verifier_dppo_math_fixed_dapo_100step_4node__42__1788675225}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$RUN_MODE" == "smoke" ]]; then
    export EXP_NAME="${EXP_NAME:-qwen35_2b_opd_from_verifier_2b_fixed_batches_smoke_1node}"
    # The smoke recipe runs a single 64-episode step at async_steps=1; run four
    # batches at async_steps=2 so the parking path is exercised (4 x 32 x 2).
    FIXED_ARGS=(--async_steps 2 --fixed_prompt_batches true --total_episodes 256)
else
    export EXP_NAME="${EXP_NAME:-qwen35_2b_opd_from_verifier_2b_fixed_batches_async4_lr1e6_100step_4node}"
    FIXED_ARGS=(--fixed_prompt_batches true --save_freq 10)
fi

exec "$HERE/qwen35_2b_math_objective_control_4node.sh" \
    "$BEAKER_IMAGE" "$DAPO_SPLIT_DATASET" "${FIXED_ARGS[@]}" "$@"
