#!/bin/bash
# ==========================================================================
# End-to-end test script for adaptive rubrics (single GPU, no search tools)
#
# Prerequisites:
#   - 1x GPU with >=40 GB memory (H100, A100, etc.)
#   - OPENAI_API_KEY set for rubric scoring (or AZURE_API_KEY)
#   - Python environment with open-instruct dependencies installed
#
# Usage:
#   # 1. Set your API key:
#   export OPENAI_API_KEY=your_key
#
#   # 2. Run locally on a GPU node:
#   bash scripts/train/debug/adaptive_rubric_mini_test.sh
#
#   # 3. Or run via Beaker:
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/adaptive_rubric_mini_test.sh
#
# What this tests:
#   - GRPO training with adaptive rubric rewards enabled
#   - RubricVerifier scoring via LLM judge (gpt-4.1-mini)
#   - Rubric buffer initialization and update across steps
#   - Static rubrics as persistent rubrics
#   - Single-GPU mode with DeepSpeed stage 3 + vLLM
#
# Expected runtime: ~8-10 minutes on a single H100/A100.
# ==========================================================================

set -e

model_path=Qwen/Qwen3-0.6B
dataset_list="rl-research/dr-tulu-rl-data 1.0"
exp_name="adaptive-rubric-mini-test"

# ---- Environment setup ----
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RUBRIC_JUDGE_MODEL=gpt-4.1-mini

# Set PYTHONPATH to repo root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$AZURE_API_KEY" ]; then
    echo "WARNING: Neither OPENAI_API_KEY nor AZURE_API_KEY is set."
    echo "Rubric scoring will return 0 for all samples (training still runs)."
    echo "Set one with: export OPENAI_API_KEY=your_key"
fi

echo "=============================================="
echo "Starting Adaptive Rubric Mini Training Test"
echo "=============================================="
echo "Model: ${model_path}"
echo "Dataset: ${dataset_list}"
echo "Exp name: ${exp_name}"
echo "Repo root: ${REPO_ROOT}"
echo "=============================================="

python3 open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --wandb_project_name rl-rag \
    --beta 0.001 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --output_dir /tmp/adaptive_rubric_test_output \
    --kl_estimator 3 \
    --dataset_mixer_list ${dataset_list} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 8 \
    --dataset_mixer_eval_list_splits train \
    --apply_adaptive_rubric_reward true \
    --normalize_rubric_scores false \
    --use_rubric_buffer true \
    --use_static_rubrics_as_persistent_rubrics true \
    --max_active_rubrics 5 \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 2048 \
    --model_name_or_path ${model_path} \
    --attn_implementation sdpa \
    --non_stop_penalty False \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 50 \
    --deepspeed_stage 3 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --single_gpu_mode True \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_sync_backend gloo \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --save_freq 25 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --gradient_checkpointing \
    --max_tool_calls 0 \
    --only_reward_good_outputs False \
    2>&1 | tee /tmp/adaptive_rubric_test.log

echo "=============================================="
echo "Test completed! Log saved to /tmp/adaptive_rubric_test.log"
echo "=============================================="

