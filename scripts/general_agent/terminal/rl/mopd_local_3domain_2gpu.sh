#!/bin/bash

# Local 2-GPU MOPD test with THREE routed domains in one batch:
#   terminal (swerl sandbox, text env)  -> teacher 0 allenai/tmax-4b     (tag "passthrough")
#   math     (verifier only)           -> teacher 1 Qwen/Qwen3.5-4B     (tag "math")
#   guess_number (in-process tool env) -> teacher 2 Qwen/Qwen3.5-0.8B   (tag "guess_number")
#
# Teacher 2 is the student itself, so its routed KL must be ~0 — a built-in
# correctness sentinel for K=3 routing. Notes:
# - Every rollout resets one actor from EVERY configured pool, so every row
#   carries a union env_config (nested env_configs form) configuring both
#   swerl_sandbox (per-task image, or slim stub) and guess_number (secret).
# - guess rows are re-tagged "guess_number" so routing can tell them apart
#   from terminal's "passthrough"; their env reward is skipped (fine under
#   pure OPD). wordle is excluded: only one text env per rollout.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

if [ -z "${DOCKER_HOST:-}" ] && [ ! -S /var/run/docker.sock ]; then
    echo "No host docker daemon; starting podman via scripts/docker/docker_login.sh"
    export PODMAN_LOG_DIR="${PODMAN_LOG_DIR:-/tmp/podman-logs}"
    source scripts/docker/docker_login.sh
fi

DATA_DIR="output/mopd_3domain"
if [ ! -f "$DATA_DIR/terminal.jsonl" ]; then
    mkdir -p "$DATA_DIR"
    uv run --active python - "$DATA_DIR" <<'EOF'
import sys
from datasets import load_dataset

out = sys.argv[1]
SLIM = "python:3.12-slim"

def union_env_config(swerl_image, swerl_task_id, guess_number):
    # One schema for every row of every derived set, so the transformed
    # datasets concatenate cleanly and both pools' resets get their kwargs.
    return {
        "env_configs": [
            {"env_name": "swerl_sandbox", "image": swerl_image, "task_id": swerl_task_id, "number": ""},
            {"env_name": "guess_number", "image": "", "task_id": "", "number": guess_number},
        ]
    }

term = load_dataset("hamishivi/swerl-tmax-10k", split="train").select(range(64))
term = term.map(
    lambda ex: {"env_config": union_env_config(ex["env_config"]["image"], ex["env_config"]["task_id"], "1")}
)
term.to_json(f"{out}/terminal.jsonl")

math = load_dataset("ai2-adapt-dev/rlvr_open_reasoner_math", split="train").select(range(64))
math = math.map(lambda ex: {"env_config": union_env_config(SLIM, "", "1")})
math.to_json(f"{out}/math.jsonl")

guess = load_dataset("hamishivi/rlenv-guess-number", split="train")
guess = guess.map(
    lambda ex: {
        "dataset": "guess_number",
        "env_config": union_env_config(SLIM, "", ex["env_config"]["number"]),
    }
)
guess.to_json(f"{out}/guess.jsonl")
print("Wrote 3-domain derived datasets to", out)
EOF
fi

uv run --active python open_instruct/grpo_fast.py \
    --exp_name mopd_local_3domain_2gpu \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_mixer_list output/mopd_3domain/terminal.jsonl 24 output/mopd_3domain/math.jsonl 24 output/mopd_3domain/guess.jsonl 24 \
    --dataset_mixer_list_splits train \
    --opd_teacher_model_name_or_path allenai/tmax-4b Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B \
    --opd_teacher_combine route \
    --opd_teacher_domains passthrough math guess_number \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 6 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 48 \
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
    --tools swerl_sandbox guess_number \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 60, "image": "python:3.12-slim"}' '{}' \
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
