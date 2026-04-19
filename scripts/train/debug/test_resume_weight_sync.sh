#!/bin/bash
# Tests that weight sync fires correctly when resuming training.
# Runs phase 1 (1 step + checkpoint), then phase 2 (resume), and verifies
# "[Main Thread] Initializing native vLLM weight sync." appears in phase 2 logs.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
unset LD_LIBRARY_PATH
export HF_HUB_CACHE=/tmp/hf_hub_cache_resume_test
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_resume_test
# Unset BEAKER_JOB_ID so is_beaker_job() returns False and the dataset cache
# uses a local writable path instead of the read-only /weka path.
unset BEAKER_JOB_ID

CKPT_DIR="/tmp/grpo_resume_weight_sync_test_ckpt"
OUTPUT_DIR="/tmp/grpo_resume_weight_sync_test_out"
LOG1="/tmp/grpo_resume_phase1.log"
LOG2="/tmp/grpo_resume_phase2.log"

# num_training_steps = total_episodes / (num_unique_prompts_rollout * num_samples_per_prompt_rollout)
# = 32 / (8 * 4) = 1  (phase 1)
# = 64 / (8 * 4) = 2  (phase 2, so resume_training_step=2 and loop runs once)

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
    --num_learners_per_node 1
    --vllm_tensor_parallel_size 1
    --beta 0.01
    --seed 3
    --local_eval_every -1
    --vllm_sync_backend gloo
    --vllm_gpu_memory_utilization 0.4
    --vllm_enforce_eager
    --single_gpu_mode
    --push_to_hub false
    --output_dir "$OUTPUT_DIR"
    --checkpoint_state_dir "$CKPT_DIR"
    --checkpoint_state_freq 1
    --deepspeed_stage 2
)

rm -rf "$CKPT_DIR" "$OUTPUT_DIR" "$LOG1" "$LOG2"

PYTHON=/root/calzone/open-instruct/.venv/bin/python

echo "=== Phase 1: 1 training step (saves checkpoint at step 1) ==="
"$PYTHON" open_instruct/grpo_fast.py \
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
echo "=== Phase 2: Resume from checkpoint (should init weight sync at resume step) ==="
"$PYTHON" open_instruct/grpo_fast.py \
    "${COMMON_ARGS[@]}" \
    --total_episodes 64 \
    2>&1 | tee "$LOG2"

if grep -q "\[Main Thread\] Initializing native vLLM weight sync" "$LOG2"; then
    echo "✅ Phase 2: weight sync initialized on resume — fix confirmed!"
else
    echo "❌ Phase 2: weight sync NOT initialized on resume — fix failed"
    exit 1
fi

echo ""
echo "=== All checks passed ==="
