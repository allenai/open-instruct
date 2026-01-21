#!/bin/bash
# =============================================================================
# FP32 LM Head Comparison: 4 Sequential Runs
# =============================================================================
#
# Runs 4 experiments overnight to compare fp32 modes:
#   1. No FP32 (baseline)
#   2. FP32 cache mode (--fp32_lm_head)
#   3. FP32 permanent mode (--fp32_lm_head --fp32_lm_head_permanent)
#   4. No FP32 (baseline repeat for variance check)
#
# All runs use identical hyperparameters matching the original no_fp32 run.
#
# Usage:
#   ./scripts/train/dgx-spark/fp32_comparison_overnight.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "FP32 LM Head Comparison - Overnight Runs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Environment setup
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Common hyperparameters (matching the no_fp32 baseline)
MODEL="Qwen/Qwen2.5-0.5B"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
DATASET_SIZE=5000
TOTAL_EPISODES=20000
BATCH_SIZE=4
NUM_PROMPTS=16
NUM_SAMPLES=4
RESPONSE_LENGTH=512
PACK_LENGTH=768
LEARNING_RATE=1e-6
SEED=42

run_experiment() {
    local exp_name=$1
    local fp32_flags=$2
    local output_dir="/tmp/grpo_fp32_comparison_${exp_name}"

    echo ""
    echo "=============================================="
    echo "Starting: $exp_name"
    echo "FP32 flags: ${fp32_flags:-none}"
    echo "Output: $output_dir"
    echo "Time: $(date)"
    echo "=============================================="

    # Clean up any leftover processes
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 5

    # Check memory
    FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    echo "Available memory: ${FREE_MEM_GB}GB"

    rm -rf "$output_dir"

    uv run python open_instruct/grpo_fast.py \
        --model_name_or_path "$MODEL" \
        --dataset_mixer_list "$DATASET" $DATASET_SIZE \
        --dataset_mixer_list_splits train \
        --validation_holdout_ratio 0.1 \
        --max_prompt_token_length 256 \
        --response_length $RESPONSE_LENGTH \
        --pack_length $PACK_LENGTH \
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
        --beta 0.0 \
        --load_ref_policy false \
        $fp32_flags \
        --save_logprob_samples true \
        --save_logprob_samples_max 50000 \
        --seed $SEED \
        --local_eval_every 10 \
        --save_freq 100 \
        --single_gpu_mode \
        --vllm_sync_backend gloo \
        --vllm_gpu_memory_utilization 0.5 \
        --vllm_enforce_eager \
        --gradient_checkpointing \
        --attn_implementation sdpa \
        --output_dir "$output_dir" \
        --with_tracking \
        --push_to_hub false \
        --exp_name "$exp_name"

    echo ""
    echo "Completed: $exp_name at $(date)"
    echo ""

    # Brief pause between runs
    sleep 10
}

# Run 1: No FP32 (baseline)
run_experiment "fp32_comparison_no_fp32_baseline" ""

# Run 2: FP32 cache mode
run_experiment "fp32_comparison_cache_mode" "--fp32_lm_head true"

# Run 3: FP32 permanent mode
run_experiment "fp32_comparison_permanent_mode" "--fp32_lm_head true --fp32_lm_head_permanent true"

# Run 4: No FP32 (repeat for variance)
run_experiment "fp32_comparison_no_fp32_repeat" ""

echo ""
echo "=============================================="
echo "All experiments completed at: $(date)"
echo "=============================================="
