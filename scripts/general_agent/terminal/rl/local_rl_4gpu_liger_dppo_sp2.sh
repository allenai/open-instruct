#!/bin/bash

# Local 4-GPU validation run for the prod DPPO loss path (tiled/liger GRPO loss + DPPO +
# ZeRO-3 + Ulysses SP=2) -- the exact code path touched by the loss-scale fix
# (tiled_grpo_loss_scale / deepspeed_gradient_reduction_divisor).
# Layout: 2 learner GPUs (SP=2) + 2 vLLM engine GPUs. Qwen3-0.6B + small tmax slice.
# Same as local_rl_4gpu.sh plus the prod loss flags. 5 training steps (160 episodes / 32).
#
# Usage: EXP_SUFFIX=fixed ./scripts/general_agent/terminal/rl/local_rl_4gpu_liger_dppo_sp2.sh
#   PYTHON_BIN  - python to run the driver with (default: `uv run python`). Use an existing
#                 venv python (e.g. /stage/.venv/bin/python) to avoid a fresh uv env resolve.
#   EXP_SUFFIX  - appended to the exp_name so fixed/baseline runs are distinguishable in wandb.

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
# Ray re-wraps workers in `uv run` otherwise, which fails inside the uv cache dir.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Container runtime for the swerl_sandbox tool. On the podman sandbox VM (no host docker),
# bring up an in-session podman system service and point the SDK at it via DOCKER_HOST.
if [ -z "$DOCKER_HOST" ] && [ ! -S /var/run/docker.sock ]; then
    echo "No host docker daemon; starting podman via scripts/docker/docker_login.sh"
    export PODMAN_LOG_DIR="${PODMAN_LOG_DIR:-/tmp/podman-logs}"
    source scripts/docker/docker_login.sh   # uses DOCKER_PAT from the sandbox session's secret-env
fi

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
EXP_SUFFIX="${EXP_SUFFIX:-run}"

$PYTHON_BIN open_instruct/grpo_fast.py \
    --exp_name terminal_local_rl_liger_dppo_sp2_${EXP_SUFFIX} \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 32 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates true \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 160 \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --lm_head_fp32 true \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 60, "image": "python:3.12-slim"}' \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --pool_size 32 \
    --max_steps 10 \
    --backend_timeout 300 \
    --gradient_checkpointing \
    --save_traces \
    --local_eval_every 8 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project oe-general-agents \
    --output_dir output/tmax_rl_local_4gpu_liger_dppo_sp2_${EXP_SUFFIX} \
    --push_to_hub false
