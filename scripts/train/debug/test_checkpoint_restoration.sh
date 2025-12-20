#!/bin/bash
# Test that num_total_tokens is correctly restored from checkpoint.
# This test runs full GRPO training twice:
# 1. First run: train for 1 step, checkpoint
# 2. Second run: resume from checkpoint
# 3. Verify num_total_tokens was restored

set -e

TMPDIR=$(mktemp -d)
CHECKPOINT_DIR="$TMPDIR/checkpoints"
OUTPUT_DIR="$TMPDIR/output"

cleanup() {
    rm -rf "$TMPDIR"
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Test: Checkpoint num_total_tokens restoration ==="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Output dir: $OUTPUT_DIR"

source configs/beaker_configs/ray_node_setup.sh

COMMON_ARGS=(
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 16
    --dataset_mixer_list_splits train
    --max_prompt_token_length 128
    --response_length 32
    --pack_length 256
    --per_device_train_batch_size 1
    --num_unique_prompts_rollout 2
    --num_samples_per_prompt_rollout 4
    --model_name_or_path Qwen/Qwen2.5-0.5B
    --stop_strings "</answer>"
    --apply_r1_style_format_reward
    --apply_verifiable_reward true
    --temperature 1.0
    --ground_truths_key ground_truth
    --chat_template_name r1_simple_chat_postpend_think
    --learning_rate 3e-7
    --deepspeed_stage 2
    --num_learners_per_node 1
    --vllm_tensor_parallel_size 1
    --beta 0.0
    --seed 3
    --vllm_sync_backend gloo
    --vllm_gpu_memory_utilization 0.3
    --vllm_enforce_eager
    --gradient_checkpointing
    --single_gpu_mode
    --filter_zero_std_samples False
    --checkpoint_state_freq 1
    --checkpoint_state_dir "$CHECKPOINT_DIR"
    --output_dir "$OUTPUT_DIR"
)

echo ""
echo "=== First run: Train for 1 step ==="
python open_instruct/grpo_fast.py \
    "${COMMON_ARGS[@]}" \
    --num_training_steps 1 \
    2>&1 | tee /tmp/grpo_run1.log

EXPECTED_TOKENS=$(grep -oP 'num_total_tokens.*?(\d+)' /tmp/grpo_run1.log | grep -oP '\d+' | tail -1)

if [ -z "$EXPECTED_TOKENS" ] || [ "$EXPECTED_TOKENS" -eq 0 ]; then
    echo "ERROR: Could not find num_total_tokens in first run logs"
    exit 1
fi

echo ""
echo "=== First run completed. num_total_tokens = $EXPECTED_TOKENS ==="
echo ""

ray stop --force
sleep 5

source configs/beaker_configs/ray_node_setup.sh

echo ""
echo "=== Second run: Resume from checkpoint ==="
python open_instruct/grpo_fast.py \
    "${COMMON_ARGS[@]}" \
    --num_training_steps 2 \
    2>&1 | tee /tmp/grpo_run2.log

RESTORED_TOKENS=$(grep -oP 'Restored num_total_tokens: (\d+)' /tmp/grpo_run2.log | grep -oP '\d+' | tail -1)

if [ -z "$RESTORED_TOKENS" ]; then
    echo ""
    echo "=== TEST FAILED ==="
    echo "ERROR: num_total_tokens was NOT restored from checkpoint."
    echo "Expected 'Restored num_total_tokens: $EXPECTED_TOKENS' in logs."
    exit 1
fi

if [ "$RESTORED_TOKENS" -ne "$EXPECTED_TOKENS" ]; then
    echo ""
    echo "=== TEST FAILED ==="
    echo "ERROR: num_total_tokens mismatch."
    echo "Expected: $EXPECTED_TOKENS"
    echo "Got: $RESTORED_TOKENS"
    exit 1
fi

echo ""
echo "=== TEST PASSED ==="
echo "num_total_tokens correctly restored: $RESTORED_TOKENS (expected: $EXPECTED_TOKENS)"
exit 0
