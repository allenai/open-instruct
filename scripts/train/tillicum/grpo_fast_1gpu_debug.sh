#!/bin/bash
# GRPO Fast 1-GPU Debug Script for Tillicum
#
# This script runs a small GRPO training job on a single GPU using the debug QOS.
# Perfect for testing that your setup works before scaling up.
#
# Usage:
#   bash scripts/train/tillicum/grpo_fast_1gpu_debug.sh
#
# Or with dry-run to preview:
#   bash scripts/train/tillicum/grpo_fast_1gpu_debug.sh --dry_run
#
# Note: Uses cuda/13.0.0 which requires gcc/13.4.0 on Tillicum.
# Check available modules with `module avail cuda`
#
# IMPORTANT: Update SCRATCH_DIR to your scratch space on Tillicum!
SCRATCH_DIR="${SCRATCH_DIR:-/gpfs/scrubbed/$USER}"

uv run python tillicum.py \
    --qos debug \
    --gpus 1 \
    --time 01:00:00 \
    --job_name grpo_debug \
    --module gcc/13.4.0 \
    --module cuda/13.0.0 \
    "$@" \
    -- \
    python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --rollouts_save_path "${SCRATCH_DIR}/rollouts/" \
    --output_dir "${SCRATCH_DIR}/output/" \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --system_prompt_override_file scripts/train/debug/cute_debug_system_prompt.txt \
    --active_sampling --async_steps 8
