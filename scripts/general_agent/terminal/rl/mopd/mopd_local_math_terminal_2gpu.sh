#!/bin/bash

# Local 2-GPU MOPD test with a REAL capability mix: Math + Terminal.
#
# Student Qwen3.5-0.8B rolls out on a mixed batch of terminal sandbox tasks
# (hamishivi/swerl-tmax-10k, dataset tag "passthrough", swerl_sandbox tool in
# local docker/podman) and math problems (ai2-adapt-dev/rlvr_open_reasoner_math,
# tag "math"). MOPD `route` sends every terminal token to the terminal-RL
# teacher allenai/tmax-4b and every math token to Qwen/Qwen3.5-4B — the MOPD
# paper's per-domain-specialist setup in miniature. Pure OPD (rewards logged
# only). Layout: GPU 0 learner (student + both 4B teachers), GPU 1 vLLM engine.
#
# Watch objective/opd_route_frac_teacher_{0,1} (both > 0 on mixed steps) and
# the per-teacher KLs. Based on local_rl_2gpu.sh; docker must work locally
# (podman fallback below).
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
# Ray re-wraps workers in `uv run` otherwise, which can resolve to a
# cache-managed env and refuse to start.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Container runtime for the swerl_sandbox tool (see local_rl_2gpu.sh).
if [ -z "${DOCKER_HOST:-}" ] && [ ! -S /var/run/docker.sock ]; then
    echo "No host docker daemon; starting podman via scripts/docker/docker_login.sh"
    export PODMAN_LOG_DIR="${PODMAN_LOG_DIR:-/tmp/podman-logs}"
    source scripts/docker/docker_login.sh
fi

# Derived math dataset: every rollout resets one actor from EVERY configured
# pool (multi_task_rl.md caveat 1), and SWERLSandboxEnv.reset refuses samples
# without an explicit image. Give the math rows a stub env_config pointing at
# the cheap slim image (schema-matched to the swerl rows, task_id empty) so
# their sandbox resets succeed and sit idle while the math verifier scores them.
MATH_JSONL="output/math_rlvr_with_swerl_envcfg.jsonl"
if [ ! -f "$MATH_JSONL" ]; then
    mkdir -p output
    uv run --active python - "$MATH_JSONL" <<'EOF'
import sys
from datasets import load_dataset

d = load_dataset("ai2-adapt-dev/rlvr_open_reasoner_math", split="train").select(range(64))
d = d.map(lambda ex: {"env_config": {"env_name": "swerl_sandbox", "image": "python:3.12-slim", "task_id": ""}})
d.to_json(sys.argv[1])
print(f"Wrote {len(d)} math rows with stub env_config to {sys.argv[1]}")
EOF
fi

uv run --active python open_instruct/grpo_fast.py \
    --exp_name mopd_local_math_terminal_2gpu \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 32 output/math_rlvr_with_swerl_envcfg.jsonl 32 \
    --dataset_mixer_list_splits train \
    --opd_teacher_model_name_or_path allenai/tmax-4b Qwen/Qwen3.5-4B \
    --opd_teacher_combine route \
    --opd_teacher_domains passthrough math \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 32 \
    --deepspeed_stage 3 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --load_ref_policy false \
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
    --local_eval_every -1 \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false "$@"
