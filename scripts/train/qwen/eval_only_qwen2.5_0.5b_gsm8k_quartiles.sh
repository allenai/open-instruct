#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen25_05b_it_eval_only_gsm8k_quartiles}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
BEAKER_IMAGE="michaeln/open_instruct"

# Training data is unused in eval-only mode but still required by config parsing.
DATASETS="${DATASETS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-test}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/prometheus \
    --workspace ai2/oe-adapt-code \
    --priority low \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus 1 \
    --budget ai2/oe-adapt \
    -- \
uv run --active open_instruct/grpo_fast.py \
    --output_dir results \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --beta 0.0 \
    --async_steps 1 \
    --inflight_updates \
    --filter_zero_std_samples False \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ${DATASETS} \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --dataset_mixer_eval_list ${LOCAL_EVALS} \
    --dataset_mixer_eval_list_splits "${LOCAL_EVAL_SPLITS}" \
    --max_prompt_token_length 512 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --chat_template_name qwen_instruct_boxed_math \
    --non_stop_penalty False \
    --apply_verifiable_reward true \
    --vllm_enable_prefix_caching \
    --vllm_num_engines 1 \
    --eval_on_step_0 \
    --eval_only \
    --eval_temperature 1.0 \
    --eval_top_p 1.0 \
    --eval_pass_at_k 128 \
    --with_tracking \
    --push_to_hub False "$@"
