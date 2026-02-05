#!/bin/bash
# Mini test script for adaptive rubrics without search tools
# Run on H100: ssh h100-xxx "cd /checkpoint/comem/rulin/open-instruct && bash scripts/train/debug/adaptive_rubric_mini_test.sh"

set -e

model_path=Qwen/Qwen3-0.6B
dataset_list="rl-research/dr-tulu-rl-data 1.0"
exp_name="adaptive-rubric-mini-test"

# Environment setup
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RUBRIC_JUDGE_MODEL=gpt-4.1-mini
export PYTHONPATH=/checkpoint/comem/rulin/open-instruct:$PYTHONPATH
# Disable flash attention to avoid incompatibility
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_V1=0

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set - rubric scoring will fail"
    echo "Set it with: export OPENAI_API_KEY=your_key"
fi

echo "=============================================="
echo "Starting Adaptive Rubric Mini Training Test"
echo "=============================================="
echo "Model: ${model_path}"
echo "Dataset: ${dataset_list}"
echo "Exp name: ${exp_name}"
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

