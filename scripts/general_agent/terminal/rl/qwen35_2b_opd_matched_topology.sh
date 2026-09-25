#!/usr/bin/env bash
# Qwen3.5-2B OPD comparison on one Holmes 8xB300 node. The paired MILES
# configs live in the campaign scratch directory; this script is intentionally
# explicit about the legacy stack's four learner/four vLLM GPU allocation.
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: MODE=smoke|sync|async $0 <beaker-image>}"
MODE="${MODE:?Set MODE=smoke, sync, or async}"
PATCH_DATASET="${PATCH_DATASET:?Set PATCH_DATASET to the committed Qwen math code-patch dataset}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CLUSTER=ai2/holmes
export WORKSPACE=ai2/open-instruct-dev
export BUDGET=ai2/oe-other
export PRIORITY=urgent
export OBJECTIVE=opd
export RUN_MODE=smoke  # The base recipe's one-node, 4+4 GPU topology.
export BENCH_MIN_RUNTIME=4h
export BENCH_TIMEOUT=8h
export TEACHER_MODEL=/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/qwen35_2b_verifier_dppo_math_fixed_dapo_100step_4node__42__1788675225

DAPO_SPLIT_DATASET=01M3CZ0QWJEJYVTQ0Q0NRYQJM4
COMMON_ARGS=(
    --model_revision 15852e8c16360a2fea060d615a32b45270f8a8fc
    --loss_fn dapo
    --clip_lower 0.2
    --clip_higher 0.28
    --loss_denominator token
    --num_unique_prompts_rollout 128
    --num_samples_per_prompt_rollout 2
    --response_length 16384
    --eval_response_length 16384
    --pack_length 18432
    --local_eval_every 16
    --save_freq -1
    --checkpoint_state_freq -1
    --total_episodes 4096  # 16 updates x 128 prompts x 2 responses.
)

case "$MODE" in
    smoke)
        # Qualify the CUDA 13 image, 4+4 placement, teacher and code overlay
        # cheaply before reserving full-context comparison runs.
        export EXP_NAME=opd2b-topo8b300-oi-smoke-20260925
        COMMON_ARGS+=(
            --num_unique_prompts_rollout 32
            --response_length 1024
            --eval_response_length 512
            --pack_length 3072
            --total_episodes 128
            --eval_on_step_0 false
            --local_eval_every -1
            --async_steps 1
            --inflight_updates false
            --synchronous_rollouts true
        )
        ;;
    sync)
        export EXP_NAME=opd2b-topo8b300-oi-sync-16u-20260925
        COMMON_ARGS+=(
            --async_steps 1
            --inflight_updates false
            --synchronous_rollouts true
        )
        ;;
    async)
        export EXP_NAME=opd2b-topo8b300-oi-async-16u-20260925
        COMMON_ARGS+=(
            --async_steps 3
            --inflight_updates false
            --fixed_prompt_batches true
            --synchronous_rollouts false
        )
        ;;
    *)
        echo "MODE must be smoke, sync, or async; got $MODE" >&2
        exit 2
        ;;
esac

export RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date -u +%Y%m%dT%H%M%SZ)}"
exec "$HERE/qwen35_2b_math_objective_control_4node.sh" \
    "$BEAKER_IMAGE" "$DAPO_SPLIT_DATASET" "${COMMON_ARGS[@]}"
