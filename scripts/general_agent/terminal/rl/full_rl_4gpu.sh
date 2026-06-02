#!/bin/bash

# Full-scale terminal RL on 4 GPUs — no Beaker/mason.py.
# Layout: 2 learner GPUs (SP=2) + 2 vLLM engine GPUs = 4 GPUs total.
# Same ratio as the 8-GPU and production configs.
#
# Requirements:
#   - Podman available (standard on Jupiter nodes)
#   - DOCKER_PAT env var set for Docker Hub pulls
#   - DOCKERHUB_USERNAME set (default: shashankg209)

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export REPO_PATH="${REPO_PATH:-$(pwd)}"
export PYTHONPATH="$REPO_PATH"
export PATH="/root/.local/bin:$PATH"
export DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-shashankg209}"
export MIRROR_URL="${MIRROR_URL:-jupiter-cs-aus-150.reviz.ai2.in:5000}"

export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
export SWERL_PODMAN_SERVICE_COUNT=4
export SWERL_DOCKER_START_CONCURRENCY=64
export SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0
export PODMAN_NUM_LOCKS=65536
export CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf

mkdir -p "$HOME/.triton/autotune"

source scripts/docker/docker_login.sh

ray stop --force
ray start --head --port=8888 --dashboard-host=0.0.0.0

uv run python open_instruct/grpo_fast.py \
    --exp_name terminal_local_rl_qwen35_4b_tmax_10k_4gpu \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates true \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 1280 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --verification_reward 1.0 \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --pool_size 32 \
    --max_steps 100 \
    --backend_timeout 1200 \
    --gradient_checkpointing \
    --save_traces \
    --local_eval_every 10 \
    --save_freq 20 \
    --checkpoint_state_freq 10 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --output_dir output/tmax_rl_4gpu_local \
    --push_to_hub false
