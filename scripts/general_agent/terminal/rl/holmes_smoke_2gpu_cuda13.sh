#!/bin/bash

# 2-GPU Terminal RL SMOKE TEST on ai2/holmes (CUDA 13 / B300).
# Purpose: validate the cu13 image + holmes cluster + terminal-RL pipeline end to end
# on Beaker. Same tiny config validated locally in local_rl_2gpu.sh (Qwen3-0.6B,
# swerl_sandbox, ~2 training steps) -- NOT a real training run.
#
# Layout: 1 node x 2 GPUs = 1 learner GPU + 1 vLLM engine GPU (no sequence parallelism).
# Launch via:  ./scripts/train/build_image_and_launch_dirty.sh --cuda-version 13 \
#                  scripts/general_agent/terminal/rl/holmes_smoke_2gpu_cuda13.sh
#
# Submit-side note: `uv run` pins the cuda13 group so mason.py runs on a CUDA-13 box
# (a bare `uv run` defaults to the cuda12 group and would fail building cu128 locally).

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
MODEL=Qwen/Qwen3-0.6B
TOKENIZER=Qwen/Qwen3-0.6B
DATASET=hamishivi/swerl-tmax-10k

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "CUDA-13/B300 2-GPU terminal RL smoke test (Qwen3-0.6B, swerl_sandbox)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 1 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=2 \
       --env SWERL_DOCKER_START_CONCURRENCY=32 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_PODMAN_IMAGE_JANITOR_ENABLED=1 \
       --env SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S=60 \
       --env SWERL_PODMAN_IMAGE_JANITOR_UNTIL=10m \
       --env MIRROR_URL=jupiter-cs-aus-112.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 2 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name terminal_holmes_smoke_2gpu_cuda13 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
    --dataset_mixer_list $DATASET 32 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 32 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 60, "image": "python:3.12-slim"}' \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --pool_size 16 \
    --max_steps 10 \
    --backend_timeout 300 \
    --gradient_checkpointing \
    --save_traces \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --local_eval_every 8 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project oe-general-agents \
    --output_dir /output \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs_on_weka False
