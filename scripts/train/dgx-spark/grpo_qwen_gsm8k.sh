#!/bin/bash
# =============================================================================
# GRPO Training: Qwen2.5-0.5B on GSM8K with Validation Reward Tracking
# =============================================================================
#
# Purpose: Test validation reward tracking with a larger model
# Model: Qwen/Qwen2.5-0.5B (500M params - base model, not instruct)
# Dataset: ai2-adapt-dev/rlvr_gsm8k_zs (GSM8K math problems)
# Hardware: DGX Spark (GB10 Blackwell, 128GB unified memory)
#
# Expected memory: ~20-25GB total
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "DGX Spark GRPO Training - Qwen2.5-0.5B"
echo "With Validation Reward Tracking"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Pre-flight memory check
FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "Available memory: ${FREE_MEM_GB}GB"
if [ "$FREE_MEM_GB" -lt 40 ]; then
    echo "WARNING: Less than 40GB free. Consider cleaning up."
    echo "  pkill -9 -f 'ray::' && sleep 10"
fi
echo ""

# Environment setup for DGX Spark
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Training parameters
MODEL="Qwen/Qwen2.5-0.5B"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
TOTAL_EPISODES=20000     # 10x longer run
BATCH_SIZE=4             # Increased from 1
NUM_PROMPTS=16           # Doubled from 8
NUM_SAMPLES=4            # Per prompt
LEARNING_RATE=1e-6       # Slightly higher for faster learning
BETA=0.0                 # No KL penalty initially
VALIDATION_HOLDOUT=0.1   # 10% of training data held out for validation

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Total episodes: $TOTAL_EPISODES"
echo "  Batch size: $BATCH_SIZE"
echo "  Prompts per rollout: $NUM_PROMPTS"
echo "  Samples per prompt: $NUM_SAMPLES"
echo "  Validation holdout: ${VALIDATION_HOLDOUT} (10%)"
echo ""

# Run training
uv run python open_instruct/grpo_fast.py \
    --model_name_or_path "$MODEL" \
    --dataset_mixer_list "$DATASET" 5000 \
    --dataset_mixer_list_splits train \
    --validation_holdout_ratio $VALIDATION_HOLDOUT \
    --max_prompt_token_length 256 \
    --response_length 512 \
    --pack_length 768 \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_unique_prompts_rollout $NUM_PROMPTS \
    --num_samples_per_prompt_rollout $NUM_SAMPLES \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --filter_zero_std_samples false \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate $LEARNING_RATE \
    --total_episodes $TOTAL_EPISODES \
    --deepspeed_stage 0 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta $BETA \
    --load_ref_policy false \
    --fp32_lm_head \
    --seed 42 \
    --local_eval_every 10 \
    --save_freq 100 \
    --single_gpu_mode \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --output_dir /tmp/grpo_qwen_gsm8k \
    --with_tracking \
    --push_to_hub false \
    --exp_name dgx_spark_grpo_qwen_gsm8k

echo ""
echo "=============================================="
echo "Training complete at: $(date)"
echo "=============================================="
