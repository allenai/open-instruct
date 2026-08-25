#!/bin/bash

# Local 4-GPU smoke: REGULAR terminal RL (no OPD) on the union of THREE
# terminal datasets — allenai/open-instruct-termigen +
# allenai/open-instruct-endless-terminals + allenai/tmax-15k-open-instruct —
# in one run (the "single mixed training" baseline for terminal MOPD).
#
# All three use swerl_vanillux_sandbox and the same env_config schema, and all
# keep their "passthrough" tag so env rewards work unchanged. The one thing a
# plain mix can't do is task-data lookup across three repos (one tool pool =
# one task_data source), so a merged local task_data dir is built from the
# three repos' extracted task-data.tar.gz trees (task ids verified disjoint)
# and passed via tool_configs.task_data_dir.
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

TASK_DATA_DIR="$(pwd)/output/terminal3_task_data"
if [ ! -f "$TASK_DATA_DIR/.merge_complete" ]; then
    uv run --active python - "$TASK_DATA_DIR" <<'EOF'
import os
import sys

from open_instruct.environments.swerl_sandbox import SWERLSandboxEnv

merged = sys.argv[1]
os.makedirs(merged, exist_ok=True)
count = 0
for repo in [
    "allenai/open-instruct-termigen",
    "allenai/open-instruct-endless-terminals",
    "allenai/tmax-15k-open-instruct",
]:
    tree = SWERLSandboxEnv.resolve_task_data_dir(repo)
    for name in os.listdir(tree):
        src = os.path.join(tree, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(merged, name)
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        count += 1
with open(os.path.join(merged, ".merge_complete"), "w") as f:
    f.write("ok\n")
print(f"Merged {count} task dirs into {merged}")
EOF
fi

uv run --active python open_instruct/grpo_fast.py \
    --exp_name terminal3_mix_local_smoke_2gpu \
    --model_name_or_path allenai/tmax-2b \
    --dataset_mixer_list allenai/open-instruct-termigen 24 allenai/open-instruct-endless-terminals 24 allenai/tmax-15k-open-instruct 24 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 6 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 48 \
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
    --tool_configs "{\"task_data_dir\": \"$TASK_DATA_DIR\", \"test_timeout\": 60, \"image\": \"python:3.12-slim\"}" \
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
