#!/bin/bash
# Multi-task RL — Approach (1): single run, ONE skill per training step.
#
# !!! NOT YET RUNNABLE. !!!
#
# This script captures the launch command you'd use once MultiSkillDataLoader exists
# (see scripts/multi_task_rl/README.md for the sketch). The CLI flag --homogeneous_batches
# is a placeholder for the new flag that would activate per-skill sampling in grpo_fast.py.
#
# Why homogeneous batches? See docs/algorithms/multi_task_rl.md §2 — they let you mix a
# heavy text env (swerl_sandbox) with cheap function-calling skills WITHOUT hitting:
#   - "every rollout acquires every pool" (vllm_utils.py:943)
#   - "only one text env per rollout" (vllm_utils.py:970)
# because each step only activates one skill's pools.
#
# Dataset rows: same unified schema as the heterogeneous scripts (see §5 of the doc).

EXP_NAME="${EXP_NAME:-multi_task_homogeneous_3skills}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-4B}"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS=(
    "rl-research/dr-tulu-rl-data" "0.05"
    "hamishivi/swerl-tmax-10k" "0.05"
    # "<your-org>/<your-mcp-rl-dataset>" "1.0"
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
    --env REPO_PATH=/stage \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 \
    --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --env VLLM_USE_V1=1 \
    --env DOCKERHUB_USERNAME=shashankg209 \
    --env SWERL_PODMAN_SERVICE_COUNT=8 \
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
    \
    `# ----- PLACEHOLDER FLAG: not yet implemented. -----` \
    `# When implemented, this tells grpo_fast.py to use MultiSkillDataLoader, which` \
    `# yields prompts from a single skill per step (round-robin by default).` \
    --homogeneous_batches true \
    --homogeneous_batch_schedule round_robin \
    `# Optional: weighted sampling instead of pure round-robin.` \
    `# --homogeneous_batch_weights dr_tulu=1.0,swerl_sandbox=1.0,mcp_skill=1.0` \
    \
    --dataset_mixer_list "${DATASETS[@]}" \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 10240 \
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
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    \
    `# All tools registered. Per-row tools[] + per-step skill filter together restrict each rollout's tool set.` \
    --tools serper_search jina_browse s2_search swerl_sandbox \
    --tool_call_names google_search browse_webpage snippet_search swerl_sandbox \
    --tool_configs '{}' '{}' '{}' '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 128 \
    --max_steps 50 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --apply_evolving_rubric_reward true \
    --remap_verifier general_rubric=rubric \
    --max_active_rubrics 5 \
    --tool_parser_type vllm_qwen3_xml \
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
