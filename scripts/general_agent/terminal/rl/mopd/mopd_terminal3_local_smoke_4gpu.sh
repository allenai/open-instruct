#!/bin/bash

# Local 4-GPU smoke: MULTI-TEACHER OPD from three TERMINAL EXPERT teachers —
# the per-domain-specialist MOPD setup on real terminal data:
#   termigen rollouts  -> teacher 0 allenai/qwen35-9b-termigen
#   endless rollouts   -> teacher 1 allenai/qwen35-9b-endless
#   tmax rollouts      -> teacher 2 allenai/tmax-9b
#
# Student allenai/tmax-2b (terminal-RL'd, so rollouts solve some tasks; a 9B
# student needs the 4-node recipe), 3 ZeRO-3 learners (three 9B teachers
# shard to ~18GB/GPU + ~10GB of 2B student state) + 1 vLLM engine.
# Rollouts run in real podman sandboxes from all three task
# sets via the merged task_data dir (built by terminal3_mix_local_smoke_2gpu's
# prep, reused here). Routing needs distinguishable dataset tags, so derived
# jsonls re-tag each dataset's rows (termigen/endless/tmax); the passthrough
# env reward is thereby skipped, which is fine under --opd_pure (rewards carry
# no gradient). env_config rows are self-contained (per-task image or lookup
# by task_id) and share one schema, so no other surgery is needed.
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
DATA_DIR="output/mopd_terminal3"
if [ ! -f "$TASK_DATA_DIR/.merge_complete" ] || [ ! -f "$DATA_DIR/tmax.jsonl" ]; then
    mkdir -p "$DATA_DIR"
    uv run --active python - "$TASK_DATA_DIR" "$DATA_DIR" <<'EOF'
import os
import sys

from datasets import load_dataset

from open_instruct.environments.swerl_sandbox import SWERLSandboxEnv

merged, out = sys.argv[1], sys.argv[2]

os.makedirs(merged, exist_ok=True)
if not os.path.isfile(os.path.join(merged, ".merge_complete")):
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

# Re-tag each dataset so MOPD `route` can tell the domains apart (they all
# ship as "passthrough"). 64 rows each is plenty for a 2-step smoke.
for repo, tag in [
    ("allenai/open-instruct-termigen", "termigen"),
    ("allenai/open-instruct-endless-terminals", "endless"),
    ("allenai/tmax-15k-open-instruct", "tmax"),
]:
    d = load_dataset(repo, split="train").select(range(64))
    d = d.map(lambda ex: {"dataset": tag})
    d.to_json(f"{out}/{tag}.jsonl")
    print(f"Wrote {tag}.jsonl")
EOF
fi

uv run --active python open_instruct/grpo_fast.py \
    --exp_name mopd_terminal3_local_smoke_4gpu \
    --model_name_or_path allenai/tmax-2b \
    --dataset_mixer_list output/mopd_terminal3/termigen.jsonl 24 output/mopd_terminal3/endless.jsonl 24 output/mopd_terminal3/tmax.jsonl 24 \
    --dataset_mixer_list_splits train \
    --opd_teacher_model_name_or_path allenai/qwen35-9b-termigen allenai/qwen35-9b-endless allenai/tmax-9b \
    --opd_teacher_combine route \
    --opd_teacher_domains termigen endless tmax \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
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
    --num_learners_per_node 3 \
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
