#!/bin/bash
# =============================================================================
# GRPO Training: Qwen2.5-0.5B on GSM8K (DGX Spark)
# =============================================================================
#
# Usage:
#   FP32_LM_HEAD=1 ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh
#   FP32_LM_HEAD=0 ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh
#
# Model: Qwen/Qwen2.5-0.5B (base model)
# Dataset: ai2-adapt-dev/rlvr_gsm8k_zs
# Hardware: DGX Spark (GB10 Blackwell, unified memory)
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
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Training parameters
MODEL="Qwen/Qwen2.5-0.5B"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
TOTAL_EPISODES=20000
BATCH_SIZE=4
NUM_PROMPTS=16
NUM_SAMPLES=4
LEARNING_RATE=1e-6
BETA=0.0

FP32_LM_HEAD="${FP32_LM_HEAD:-0}"
FP32_FLAG=""
EXP_SUFFIX="no_fp32"
if [ "$FP32_LM_HEAD" = "1" ]; then
    FP32_FLAG="--fp32_lm_head"
    EXP_SUFFIX="fp32"
fi
OUTPUT_DIR="/tmp/grpo_qwen_gsm8k_${EXP_SUFFIX}"
EXP_NAME="dgx_spark_grpo_qwen_gsm8k_${EXP_SUFFIX}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Total episodes: $TOTAL_EPISODES"
echo "  Batch size: $BATCH_SIZE"
echo "  Prompts per rollout: $NUM_PROMPTS"
echo "  Samples per prompt: $NUM_SAMPLES"
echo "  FP32 LM head: ${FP32_LM_HEAD}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Run training
uv run python open_instruct/grpo_fast.py \
    --model_name_or_path "$MODEL" \
    --dataset_mixer_list "$DATASET" 5000 \
    --dataset_mixer_list_splits train \
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
    $FP32_FLAG \
    --seed 42 \
    --local_eval_every 10 \
    --save_freq 100 \
    --single_gpu_mode \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --with_tracking \
    --push_to_hub false \
    --exp_name "$EXP_NAME"

echo ""
echo "=============================================="
echo "Training complete at: $(date)"
echo "=============================================="
