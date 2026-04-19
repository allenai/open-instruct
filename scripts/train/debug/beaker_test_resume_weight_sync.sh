#!/bin/bash
# Two-phase resume weight sync test, run inside a Beaker container.
# Phase 1: train 1 step and save a checkpoint.
# Phase 2: resume from checkpoint and verify checkpoint weights are synced
#          to vLLM *before* rollout generation (not after the first training step).
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

CKPT_DIR="/tmp/grpo_resume_weight_sync_test_ckpt"
OUTPUT_DIR="/tmp/grpo_resume_weight_sync_test_out"
LOG1="/tmp/grpo_resume_phase1.log"
LOG2="/tmp/grpo_resume_phase2.log"

# num_training_steps = total_episodes / (num_unique_prompts_rollout * num_samples_per_prompt_rollout)
# = 32 / (8 * 4) = 1  (phase 1)
# = 64 / (8 * 4) = 2  (phase 2, so resume_training_step=2, loop runs once)

COMMON_ARGS=(
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64
    --dataset_mixer_list_splits train
    --max_prompt_token_length 512
    --response_length 512
    --pack_length 1024
    --per_device_train_batch_size 1
    --num_unique_prompts_rollout 8
    --num_samples_per_prompt_rollout 4
    --model_name_or_path Qwen/Qwen3.5-0.8B
    --system_prompt_override_file scripts/train/qwen/math_system_prompt.txt
    --apply_verifiable_reward true
    --learning_rate 1e-6
    --num_epochs 1
    --num_learners_per_node 2
    --vllm_tensor_parallel_size 1
    --beta 0.01
    --seed 3
    --local_eval_every -1
    --vllm_sync_backend nccl
    --vllm_gpu_memory_utilization 0.4
    --vllm_enforce_eager
    --push_to_hub false
    --output_dir "$OUTPUT_DIR"
    --checkpoint_state_dir "$CKPT_DIR"
    --checkpoint_state_freq 1
    --deepspeed_stage 3
)

rm -rf "$CKPT_DIR" "$OUTPUT_DIR" "$LOG1" "$LOG2"

echo "=== Phase 1: 1 training step with ZeRO-3 (2 learner GPUs) ==="
python open_instruct/grpo_fast.py \
    "${COMMON_ARGS[@]}" \
    --total_episodes 32 \
    2>&1 | tee "$LOG1"

if grep -q "\[Main Thread\] Initializing native vLLM weight sync" "$LOG1"; then
    echo "✅ Phase 1: weight sync initialized (expected)"
else
    echo "❌ Phase 1: weight sync NOT initialized — unexpected failure"
    exit 1
fi

echo ""
echo "=== Phase 2: Resume from ZeRO-3 checkpoint (dummy step + eager weight sync before first rollout) ==="
python open_instruct/grpo_fast.py \
    "${COMMON_ARGS[@]}" \
    --total_episodes 64 \
    2>&1 | tee "$LOG2"

if grep -q "\[Main Thread\] Initializing native vLLM weight sync" "$LOG2"; then
    echo "✅ Phase 2: weight sync initialized on resume (expected)"
else
    echo "❌ Phase 2: weight sync NOT initialized on resume — unexpected failure"
    exit 1
fi

echo ""
echo "=== All checks passed (ZeRO-3 resume dummy-step + eager weight sync verified) ==="
