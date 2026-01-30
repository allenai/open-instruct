#!/bin/bash
# GRPO Fast 2-Node (16 GPUs) Debug Script for Tillicum
#
# This script runs GRPO training using 2 Tillicum nodes (16 H200 GPUs total):
# - Node 1: 4 training GPUs + 4 inference GPUs
# - Node 2: 4 training GPUs + 4 inference GPUs
# - Total: 8 training GPUs + 8 inference GPUs
#
# Usage:
#   bash scripts/train/tillicum/grpo_fast_2node_debug.sh
#
# Or with dry-run to preview:
#   bash scripts/train/tillicum/grpo_fast_2node_debug.sh --dry_run
#
# Note: Uses cuda/13.0.0 which requires gcc/13.4.0 on Tillicum.
#
# tillicum.py automatically creates an experiment directory at:
#   /gpfs/scrubbed/$USER/experiments/{job_name}_{timestamp}_{id}/
# with subdirs: logs/, output/, checkpoints/, rollouts/

uv run python tillicum.py \
    --gpus 8 \
    --nodes 2 \
    --time 08:00:00 \
    --job_name grpo_2node_debug \
    --module gcc/13.4.0 \
    --module cuda/13.0.0 \
    "$@" \
    -- \
    python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 512 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 128 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 2 \
    --num_unique_prompts_rollout 64 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 1000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_nodes 2 \
    --num_learners_per_node 4 4 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 10 \
    --vllm_sync_backend nccl \
    --vllm_gpu_memory_utilization 0.8 \
    --save_traces \
    '--rollouts_save_path=$EXPERIMENT_DIR/rollouts' \
    '--output_dir=$EXPERIMENT_DIR/output' \
    '--dataset_local_cache_dir=$DATASET_LOCAL_CACHE_DIR' \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --system_prompt_override_file scripts/train/debug/cute_debug_system_prompt.txt \
    --active_sampling --async_steps 32
