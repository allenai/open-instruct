#!/bin/bash
# Beaker experiment for SandboxLM environment — 8 GPUs (4 learners, 4 inference engines)
#
# SandboxLM: Coding environment with execute_bash + str_replace_editor tools.
# System prompt and tools modified from https://github.com/llm-in-sandbox/llm-in-sandbox
# Uses Qwen3-4B-Instruct with 32k context for code generation tasks.
#
# Requirements:
# - Docker available in the container image
# - 8 GPUs
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/sandbox_lm_8gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SandboxLM 8-GPU (4L/4E) 32k Qwen3-4B test" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --preemptible \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 30720 \
    --pack_length 32768 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 1.0 \
    --learning_rate 5e-7 \
    --total_episodes 3200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 42 \
    --inflight_updates True \
    --async_steps 4 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --save_traces \
    --apply_verifiable_reward true \
    --tools generic_sandbox \
    --pool_size 16 \
    --max_steps 30 \
    --tool_parser_type vllm_hermes \
    --system_prompt_override_file scripts/train/debug/envs/sandbox_lm_system_prompt.txt \
    --no_filter_zero_std_samples \
    --vllm_enable_prefix_caching \
    --output_dir output/sandbox_lm_8gpu_debug
