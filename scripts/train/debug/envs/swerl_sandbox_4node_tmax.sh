#!/bin/bash
# Training script for GRPO with SWERL Sandbox environment (4 nodes x 8 GPUs on Beaker)
#
# SWERL Sandbox: Per-sample Docker tasks with bash-only tool loop (TassieAgent style).
#
# Requirements:
# - 4 nodes with 8 GPUs each (32 GPUs total, launched on Beaker)
# - Docker-in-Docker available on cluster nodes
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/swerl_sandbox_4node.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL Sandbox 4-node (32 GPU) GRPO with Qwen3-4B on tmax-skill-taxonomy" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 4 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=hamishi740 \
       --secret DOCKER_PAT=hamishivi_DOCKER_PAT \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/swerl-tmax-skill-taxonomy-1k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 100000000 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 24 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-skill-taxonomy-1k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 128 \
    --max_steps 100 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_hermes \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 50 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --no_resampling_pass_rate 0.875 \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name swerl_sandbox_qwen3_4b_4node_tmax_grpo \
    --local_eval_every 10 \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False
