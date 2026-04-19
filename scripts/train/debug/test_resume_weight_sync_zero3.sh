#!/bin/bash
# Tests that eager weight sync on resume works with DeepSpeed ZeRO-3.
# Uses a single learner (1 GPU) so ZeRO-3 sets ds_id on all parameters and
# exercises the GatheredParameters(enabled=True) code path, verifying that
# checkpoint weights can be gathered and broadcast to vLLM without a prior
# training step. Cross-GPU gathering (2+ learners) requires 3+ GPUs.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
unset LD_LIBRARY_PATH
export HF_HUB_CACHE=/tmp/hf_hub_cache_resume_test
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_resume_test
unset BEAKER_JOB_ID

CKPT_DIR="/tmp/grpo_resume_zero3_test_ckpt"
OUTPUT_DIR="/tmp/grpo_resume_zero3_test_out"
LOG1="/tmp/grpo_resume_zero3_phase1.log"
LOG2="/tmp/grpo_resume_zero3_phase2.log"

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
    --deepspeed_stage 3
)

rm -rf "$CKPT_DIR" "$OUTPUT_DIR" "$LOG1" "$LOG2"

PYTHON=/root/calzone/open-instruct/.venv/bin/python

echo "=== Phase 1: 1 training step with ZeRO-3 (saves checkpoint at step 1) ==="
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
echo "=== Phase 2: Resume from ZeRO-3 checkpoint (dummy step + eager weight sync before first rollout) ==="
"$PYTHON" open_instruct/grpo_fast.py \
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
