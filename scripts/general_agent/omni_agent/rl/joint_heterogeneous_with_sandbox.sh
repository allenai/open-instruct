#!/bin/bash
# Multi-task RL — Approach (2) extended: mix function-calling skills WITH swerl_sandbox (text env).
#
# WARNING: This script requires a small fix to open_instruct/vllm_utils.py.
# Today _acquire_and_reset_pools iterates the GLOBAL configured tool set per rollout
# (vllm_utils.py:943), so every rollout — even pure-search ones — acquires a Docker
# sandbox actor. Either:
#   (a) accept the cost (each prompt holds a sandbox for its duration), or
#   (b) patch _acquire_and_reset_pools to iterate `allowed_tools ∪ env_config.env_configs.keys()`
#       instead of `configured_tools`. See docs/algorithms/multi_task_rl.md §4 / §7.
#
# Without one of those, this run will be VERY slow on prompts that don't need the sandbox.
#
# Dataset rows for the sandbox skill should carry:
#   tools=["swerl_sandbox"]
#   env_config={"env_configs": [{"env_name": "swerl_sandbox", "task_id": "<id>"}], "max_steps": 50}
#   dataset="passthrough"   # sandbox env produces per-turn rewards; no extra verifier
#
# Dataset rows for the search skill should carry:
#   tools=["google_search", "browse_webpage", "snippet_search"]
#   env_config={"env_configs": []}
#   dataset="general_rubric"  # remapped to RubricVerifier
#
# Launch:
#   ./scripts/train/build_image_and_launch.sh scripts/multi_task_rl/joint_heterogeneous_with_sandbox.sh
#
# Adjust DATASETS to whatever you've built.

EXP_NAME="${EXP_NAME:-multi_task_joint_hetero_sandbox}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-4B}"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS=(
    "rl-research/dr-tulu-rl-data" "0.05"
    "hamishivi/swerl-tmax-10k" "0.05"
)
DATASET_SPLITS="train"

PRIORITY="${PRIORITY:-urgent}"

uv run python mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --image ${BEAKER_IMAGE} \
    --pure_docker_mode \
    --workspace ai2/general-tool-use \
    --priority ${PRIORITY} \
    --preemptible \
    --num_nodes 2 \
    --max_retries 0 \
    \
    `# Docker / Podman setup for swerl_sandbox text env.` \
    --env REPO_PATH=/stage \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 \
    --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env VLLM_USE_V1=1 \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --env DOCKERHUB_USERNAME=shashankg209 \
    --env SWERL_SANDBOX_TIMING_LOGS=1 \
    --env SWERL_DOCKER_AUTO_REMOVE=1 \
    --env SWERL_PODMAN_SERVICE_COUNT=8 \
    --env SWERL_DOCKER_START_CONCURRENCY=128 \
    --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
    --env MIRROR_URL=jupiter-cs-aus-150.reviz.ai2.in:5000 \
    --env PODMAN_NUM_LOCKS=65536 \
    --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
    --env RUBRIC_JUDGE_MODEL=gpt-4.1 \
    --env RUBRIC_GENERATION_MODEL=gpt-4.1 \
    --secret DOCKER_PAT=shashankg_DOCKER_PAT \
    --secret SERPER_API_KEY=shashankg_SERPER_API_KEY \
    --secret S2_API_KEY=shashankg_S2_API_KEY \
    --secret JINA_API_KEY=shashankg_JINA_API_KEY \
    --secret OPENAI_API_KEY=shashankg_OPENAI_API_KEY \
    --budget ai2/oe-omai \
    --mount_docker_socket \
    --gpus 8 \
    --no_auto_dataset_cache \
    -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --dataset_mixer_list "${DATASETS[@]}" \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 5120 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    \
    `# Register every tool. Per-row 'tools' column gates dispatch per prompt.` \
    `# swerl_sandbox is a text env (is_text_env=True). Only one text env may be ACTIVE per rollout,` \
    `# but having it configured + per-row gating means search rows still work cleanly.` \
    --tools serper_search jina_browse s2_search swerl_sandbox \
    --tool_call_names google_search browse_webpage snippet_search swerl_sandbox \
    --tool_configs '{}' '{}' '{}' '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    \
    --pool_size 128 \
    --max_steps 50 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    \
    `# NO --system_prompt_override_file: per-row system prompts in messages[0].` \
    \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 20 \
    --save_freq 50 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --local_eval_every 20 \
    --try_launch_beaker_eval_jobs_on_weka False
