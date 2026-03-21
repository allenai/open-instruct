#!/bin/bash

set -euo pipefail

EXP_NAME="${EXP_NAME:-olmo31_7b_rlzero_math_eval}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-allenai/Olmo-3.1-7B-RL-Zero-Math}"
MODEL_REVISION="${MODEL_REVISION:-main}"
BEAKER_IMAGE="${BEAKER_IMAGE:-nathanl/open_instruct_auto}"

DATASETS="${DATASETS:-mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-test}"
LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test}"

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-high}"
NUM_GPUS="${NUM_GPUS:-4}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
EVAL_PASS_AT_K="${EVAL_PASS_AT_K:-64}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-32768}"
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-4096}"
PACK_LENGTH="${PACK_LENGTH:-36864}"

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus "${NUM_GPUS}" \
    --budget ai2/oe-adapt \
    -- \
uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_only \
    --eval_pass_at_k "${EVAL_PASS_AT_K}" \
    --eval_temperature 1.0 \
    --eval_top_p 1.0 \
    --eval_response_length "${EVAL_RESPONSE_LENGTH}" \
    --beta 0.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ${DATASETS} \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --dataset_mixer_eval_list ${LOCAL_EVALS} \
    --dataset_mixer_eval_list_splits ${LOCAL_EVAL_SPLITS} \
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
    --response_length "${EVAL_RESPONSE_LENGTH}" \
    --pack_length "${PACK_LENGTH}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --model_revision "${MODEL_REVISION}" \
    --non_stop_penalty False \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --apply_verifiable_reward true \
    --seed 1 \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --load_ref_policy False \
    "$@"
