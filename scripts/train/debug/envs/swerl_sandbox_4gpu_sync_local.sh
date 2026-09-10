#!/bin/bash

# Local 4-GPU smoke test for SYNCHRONOUS mode (--async_steps 0) on real Terminal RL
# (GRPO + swerl_sandbox, podman sandboxes). Layout as local_rl_4gpu.sh: 2 learner
# GPUs (SP=2) + 2 vLLM engines, Qwen3-0.6B, small hamishivi/swerl-tmax-10k slice.
#
# What to verify in the logs / wandb:
#   - DataPreparationActor pushes exactly one batch per step, only after the main
#     thread's "Waiting for weight sync (synchronous mode)" timer -> release_step.
#   - model_step_mean == training_step - 1 on every step (no staleness).
#   - time/trainer_idle_waiting_for_inference ~= generation time (no overlap).
#   - stale_results_dropped == 0; no deadlock across several steps.
# active_sampling is incompatible with sync mode (asserts async_steps > 1) and
# filter_zero_std_samples is off so every batch fills without replenishment.
#
#   PYTHON=<venv python> bash scripts/train/debug/envs/swerl_sandbox_4gpu_sync_local.sh
# (PYTHON defaults to `uv run python`; on the shared-weka dev VM prefer an existing
#  venv python + PYTHONPATH=<this worktree> to avoid a uv sync / uv.lock rewrite.)

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
# ray re-wraps workers in `uv run` otherwise, which fails from a cache-managed env.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Python imports from the shared-weka venv can take minutes; the default 30s
# raylet<->runtime-env-agent registration timeout then kills the whole node
# ("Runtime Env Agent timed out in 30000ms ... raylet exited immediately").
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-600000}
# Ray 2.5x OTel metric recorder has a getenv race that can SIGSEGV env actors
# at startup (see reference_ray_otel_getenv_segfault); we read metrics via wandb.
export RAY_enable_open_telemetry=0

# Container runtime for the swerl_sandbox tool (see local_rl_4gpu.sh): no host
# docker daemon on the podman dev VM -> start podman via docker_login.sh.
if [ -z "$DOCKER_HOST" ] && [ ! -S /var/run/docker.sock ]; then
    echo "No host docker daemon; starting podman via scripts/docker/docker_login.sh"
    # Dev-VM shells run as uid 0 but with USER=<login name>; podman then assumes a
    # rootless user without subuid ranges and every container create fails with
    # "not enough unused IDs in user namespace". Force rootful podman.
    export USER=root LOGNAME=root
    export PODMAN_LOG_DIR="${PODMAN_LOG_DIR:-/tmp/podman-logs}"
    source scripts/docker/docker_login.sh
fi

PYTHON=${PYTHON:-uv run python}
EXP_NAME=${EXP_NAME:-terminal_local_sync_smoke_4gpu}

$PYTHON open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 64 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --async_steps 0 \
    --inflight_updates false \
    --filter_zero_std_samples false \
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
    --local_eval_every -1 \
    --logging_steps 1 \
    --checkpoint_state_freq 2 \
    --checkpoint_state_dir "output/${EXP_NAME}_state" \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project oe-general-agents \
    --output_dir "output/${EXP_NAME}" \
    --push_to_hub false "$@"
