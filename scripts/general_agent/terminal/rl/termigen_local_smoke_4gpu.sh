#!/bin/bash

# Local 4-GPU smoke: REGULAR terminal RL (no OPD) on the released
# allenai/open-instruct-termigen dataset (swerl_vanillux_sandbox; per-task
# images hamishi740/termigen:* resolved from the repo's task-data.tar.gz).
# allenai/tmax-2b student (terminal-RL'd, so some rollouts actually solve
# tasks and rewards are non-zero); 2 ZeRO-3 learners + 2 vLLM engines,
# 2 training steps.
# filter_zero_std_samples false keeps all-zero-reward groups so training
# steps run even when the batch solves nothing.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "${DOCKER_HOST:-}" ] && [ ! -S /var/run/docker.sock ]; then
    echo "No host docker daemon; starting podman via scripts/docker/docker_login.sh"
    export PODMAN_LOG_DIR="${PODMAN_LOG_DIR:-/tmp/podman-logs}"
    source scripts/docker/docker_login.sh
fi

DATASET="${DATASET:-allenai/open-instruct-termigen}"
EXP_NAME="${EXP_NAME:-termigen_local_smoke_2gpu}"

uv run --active python open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path allenai/tmax-2b \
    --dataset_mixer_list "$DATASET" 32 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 32 \
    --deepspeed_stage 3 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --load_ref_policy false \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --filter_zero_std_samples false \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools swerl_vanillux_sandbox \
    --tool_configs "{\"task_data_hf_repo\": \"$DATASET\", \"test_timeout\": 60, \"image\": \"python:3.12-slim\"}" \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --pool_size 16 \
    --max_steps 16 \
    --backend_timeout 300 \
    --gradient_checkpointing \
    --local_eval_every -1 \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false "$@"
