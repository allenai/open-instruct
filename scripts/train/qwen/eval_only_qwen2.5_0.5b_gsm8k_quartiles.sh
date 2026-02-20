#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen25_05b_it_eval_only_gsm8k_quartiles}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

# Training data is unused in eval-only mode but still required by config parsing.
DATASETS="${DATASETS:-mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-0 8}"
LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-0 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-25 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-50 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-75 8}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-train}"

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
    --gpus 2 \
    --budget ai2/oe-adapt \
    -- \
uv run --active open_instruct/grpo_fast.py \
    --output_dir results \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --beta 0.0 \
    --async_steps 4 \
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
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ${LOCAL_EVALS} \
    --dataset_mixer_eval_list_splits "${LOCAL_EVAL_SPLITS}" \
    --max_prompt_token_length 512 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --chat_template_name qwen_instruct_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --total_episodes 128 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 1 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --single_gpu_mode \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.7 \
    --colocate_train_inference_mode \
    --num_learners_per_node 2 \
    --vllm_tensor_parallel_size 2 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --eval_on_step_0 True \
    --eval_only True \
    --eval_pass_at_k 128 \
    --with_tracking True \
    --push_to_hub False "$@"
