#!/bin/bash
# Test script for fp32 LM head with multi-GPU tensor parallelism
# Tests larger models that require multiple GPUs per vLLM engine
#
# Default configuration (8 GPUs, 1 node):
# - 2 vLLM engines x 2 GPUs each (tensor parallel) = 4 inference GPUs
# - 4 learner GPUs for training
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/analysis/fp32-lm-head/fp32_lm_head_multigpu.sh
#
# Environment variable overrides:
#   MODEL_NAME=Qwen/Qwen2.5-14B        # Model to test (default: Qwen/Qwen2.5-7B)
#   VLLM_TP=4                          # Tensor parallel size per engine (default: 2)
#   VLLM_ENGINES=1                     # Number of vLLM engines (default: 2)
#   NUM_LEARNERS=4                     # Training GPUs per node (default: 4)
#
# Examples:
#   # 7B model (default)
#   ./scripts/train/build_image_and_launch.sh scripts/analysis/fp32-lm-head/fp32_lm_head_multigpu.sh
#
#   # MoE model
#   MODEL_NAME=Qwen/Qwen2.5-MoE-A2.7B ./scripts/train/build_image_and_launch.sh scripts/analysis/fp32-lm-head/fp32_lm_head_multigpu.sh
#
#   # 14B model with 4-way TP (1 engine, 4 learners)
#   MODEL_NAME=Qwen/Qwen2.5-14B VLLM_TP=4 VLLM_ENGINES=1 ./scripts/train/build_image_and_launch.sh scripts/analysis/fp32-lm-head/fp32_lm_head_multigpu.sh

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

# Configurable parameters via environment variables
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_ENGINES="${VLLM_ENGINES:-2}"
NUM_LEARNERS="${NUM_LEARNERS:-4}"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Testing model: $MODEL_NAME"
echo "vLLM config: $VLLM_ENGINES engines x $VLLM_TP GPUs (tensor parallel)"
echo "Training: $NUM_LEARNERS learner GPUs"
echo "Testing fp32_lm_head + fp32_lm_head_permanent mode"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "FP32 LM head multi-GPU test ($MODEL_NAME)" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority high \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 30m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --budget ai2/oe-adapt \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path "$MODEL_NAME" \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 1e-6 \
    --total_episodes 100 \
    --deepspeed_stage 3 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node "$NUM_LEARNERS" \
    --vllm_num_engines "$VLLM_ENGINES" \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    --beta 0.0 \
    --load_ref_policy false \
    --fp32_lm_head true \
    --fp32_lm_head_permanent true \
    --seed 42 \
    --local_eval_every 5 \
    --vllm_gpu_memory_utilization 0.85 \
    --gradient_checkpointing \
    --push_to_hub false \
    --exp_name fp32_lm_head_multigpu_test
