#!/bin/bash
# Debug script for testing GRPO with AppWorldEnv (train + local eval)
#
# Requirements:
# - `uv sync --extra appworld` (or `pip install appworld`)
# - `appworld install`
# - `appworld download data`
# - Datasets created via scripts/data/create_appworld_env_datasets.py

set -e

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

APPWORLD_TRAIN_DATASET="${APPWORLD_TRAIN_DATASET:-hamishivi/rlenv-appworld-train}"
APPWORLD_EVAL_DATASET="${APPWORLD_EVAL_DATASET:-hamishivi/rlenv-appworld-eval}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"

echo "Starting AppWorldEnv training (1 GPU)..."
echo "Train dataset: ${APPWORLD_TRAIN_DATASET}"
echo "Eval dataset:  ${APPWORLD_EVAL_DATASET}"
echo "Model:         ${MODEL_NAME}"

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list "${APPWORLD_TRAIN_DATASET}" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "${APPWORLD_EVAL_DATASET}" 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 3072 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path "${MODEL_NAME}" \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --total_episodes 16 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --tools appworld \
    --max_steps 40 \
    --per_turn_max_tokens 1024 \
    --tool_parser_type vllm_hermes \
    --reward_aggregator last \
    --local_eval_every 1 \
    --eval_on_step_0 true \
    --dataset_skip_cache \
    --output_dir output/appworld_debug

echo "Training complete!"
