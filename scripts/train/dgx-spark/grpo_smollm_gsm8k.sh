#!/bin/bash
# =============================================================================
# GRPO Training: SmolLM2-135M on GSM8K with Validation Reward Tracking
# =============================================================================
#
# Purpose: Test validation reward tracking feature on DGX Spark
# Model: HuggingFaceTB/SmolLM2-135M (135M params - fast iteration)
# Dataset: ai2-adapt-dev/rlvr_gsm8k_zs (GSM8K math problems)
# Hardware: DGX Spark (GB10 Blackwell, 128GB unified memory)
#
# Validation Reward Tracking:
#   This script tests the validation holdout feature. When validation_holdout_ratio > 0:
#   - Training data is split into train (90%) + validation (10%)
#   - The "eval/" metrics track accuracy on held-out validation data
#   - This allows detecting overfitting (train reward up, eval reward flat/down)
#
# Key settings for DGX Spark:
#   - single_gpu_mode: Collocate vLLM and policy on same GPU
#   - vllm_enforce_eager: Disable CUDA graphs for compatibility
#   - vllm_sync_backend gloo: Better than NCCL for single-GPU
#   - attn_implementation sdpa: SDPA faster than flash-attn on Blackwell
#   - vllm_gpu_memory_utilization 0.3: Conservative for unified memory
#
# Memory estimates (SmolLM2-135M):
#   - Model: ~0.3GB (bf16)
#   - vLLM inference: ~5GB with 0.3 utilization
#   - Training: ~2-3GB
#   - Total expected: <15GB (very safe)
#
# Usage:
#   ./scripts/train/dgx-spark/grpo_smollm_gsm8k.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "DGX Spark GRPO Training - SmolLM2-135M"
echo "With Validation Reward Tracking"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Pre-flight memory check
FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "Available memory: ${FREE_MEM_GB}GB"
if [ "$FREE_MEM_GB" -lt 30 ]; then
    echo "WARNING: Less than 30GB free. Consider cleaning up."
    echo "  pkill -9 -f 'ray::' && sleep 10"
fi
echo ""

# Environment setup for DGX Spark
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Training parameters
MODEL="HuggingFaceTB/SmolLM2-135M"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
TOTAL_EPISODES=500  # Small for initial testing
BATCH_SIZE=1
NUM_PROMPTS=4       # Per rollout
NUM_SAMPLES=4       # Per prompt
LEARNING_RATE=1e-6
BETA=0.0            # No KL penalty initially
VALIDATION_HOLDOUT=0.1  # 10% of training data held out for validation

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
    --dataset_mixer_list "$DATASET" 300 \
    --dataset_mixer_list_splits train \
    --validation_holdout_ratio $VALIDATION_HOLDOUT \
    --max_prompt_token_length 256 \
    --response_length 256 \
    --pack_length 512 \
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
    --seed 42 \
    --local_eval_every 5 \
    --save_freq 50 \
    --single_gpu_mode \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --output_dir /tmp/grpo_smollm_gsm8k \
    --with_tracking \
    --push_to_hub false \
    --exp_name dgx_spark_grpo_validation_tracking

echo ""
echo "=============================================="
echo "Training complete at: $(date)"
echo "=============================================="
