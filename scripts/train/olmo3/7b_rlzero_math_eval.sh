#!/bin/bash

set -euo pipefail

TASK_NAME="${TASK_NAME:-olmo31_7b_rlzero_math_eval}"
DESCRIPTION="${DESCRIPTION:-${TASK_NAME}_$(date +%Y%m%d_%H%M%S)}"
WORKSPACE="${WORKSPACE:-ai2/oe-adapt-code}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-high}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
BUDGET="${BUDGET:-ai2/oe-adapt}"

DEFAULT_RUN_NAME="${TASK_NAME}_$(date +%Y%m%d_%H%M%S)"

uv run mason.py \
    --task_name "${TASK_NAME}" \
    --description "${DESCRIPTION}" \
    --cluster ${CLUSTER} \
    --workspace "${WORKSPACE}" \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes "${NUM_NODES}" \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus "${NUM_GPUS}" \
    --budget "${BUDGET}" \
    -- \
uv run open_instruct/grpo_fast.py \
    --run_name "${DEFAULT_RUN_NAME}" \
    --exp_name "olmo31_7b_rlzero_math_eval" \
    --eval_only \
    --eval_pass_at_k 64 \
    --eval_temperature 1.0 \
    --eval_top_p 1.0 \
    --eval_response_length 32768 \
    --beta 0.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles 1.0 \
    --dataset_mixer_list_splits test \
    --dataset_mixer_eval_list mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles 1.0 \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length 4096 \
    --response_length 32768 \
    --pack_length 36864 \
    --model_name_or_path allenai/Olmo-3.1-7B-RL-Zero-Math \
    --model_revision main \
    --non_stop_penalty False \
    --vllm_num_engines 4 \
    --apply_verifiable_reward true \
    --seed 1 \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --push_to_hub False \
    "$@"
