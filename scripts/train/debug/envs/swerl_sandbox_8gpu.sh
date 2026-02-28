#!/bin/bash
# Training script for GRPO with SWERL Sandbox environment (8 GPUs on Beaker)
#
# SWERL Sandbox: Per-sample Docker tasks with submit-based evaluation.
# Provides execute_bash, str_replace_editor, and submit tools.
#
# Requirements:
# - 8 GPUs (launched on Beaker)
# - Docker-in-Docker available on the cluster node
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/swerl_sandbox_8gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL Sandbox 8-GPU GRPO with Qwen3-4B-Instruct" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/agent-task-combined 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 30720 \
    --pack_length 32768 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --inflight_updates true \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --total_episodes 128000 \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/agent-task-combined", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 256 \
    --max_steps 30 \
    --tool_parser_type vllm_hermes \
    --active_sampling \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name swerl_sandbox_qwen3_4b_grpo \
    --local_eval_every 10 \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False
