#!/bin/bash
# =============================================================================
# GRPO Training: Qwen2.5-0.5B on GSM8K (Short Response - Good Rewards)
# =============================================================================
#
# This configuration showed good reward scores with shorter response lengths.
# Faster training due to shorter sequences.
#
# Usage:
#   FP32_MODE=permanent ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh
#   FP32_MODE=cache ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh
#   FP32_MODE=none ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "GRPO Training - Qwen2.5-0.5B (Short Response)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Environment setup
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# FP32 mode selection
FP32_MODE="${FP32_MODE:-none}"
FP32_FLAGS=""
EXP_SUFFIX="no_fp32"

case "$FP32_MODE" in
    permanent)
        FP32_FLAGS="--fp32_lm_head true --fp32_lm_head_permanent true"
        EXP_SUFFIX="fp32_permanent"
        ;;
    cache)
        FP32_FLAGS="--fp32_lm_head true"
        EXP_SUFFIX="fp32_cache"
        ;;
    none|*)
        FP32_FLAGS=""
        EXP_SUFFIX="no_fp32"
        ;;
esac

OUTPUT_DIR="/tmp/grpo_qwen_gsm8k_short_${EXP_SUFFIX}"
EXP_NAME="grpo_qwen_gsm8k_short_${EXP_SUFFIX}"

echo "Configuration:"
echo "  FP32 mode: $FP32_MODE"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Pre-flight memory check
FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "Available memory: ${FREE_MEM_GB}GB"

rm -rf "$OUTPUT_DIR"

uv run python open_instruct/grpo_fast.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B" \
    --dataset_mixer_list "ai2-adapt-dev/rlvr_gsm8k_zs" 2000 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 256 \
    --response_length 256 \
    --pack_length 512 \
    --per_device_train_batch_size 4 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 1e-6 \
    --total_episodes 5000 \
    --deepspeed_stage 0 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy false \
    $FP32_FLAGS \
    --save_logprob_samples true \
    --save_logprob_samples_max 20000 \
    --seed 42 \
    --single_gpu_mode \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.4 \
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
