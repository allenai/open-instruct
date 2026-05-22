#!/bin/bash
# Multi-task RL — Approach (3) cascade, GENERIC STAGE N TEMPLATE.
#
# Use this when adding a third (or fourth, ...) skill onto an existing cascade.
# Copy it, fill in the SKILL_-prefixed variables, and launch.
#
# Required env vars:
#   PREV_STAGE_CHECKPOINT — final checkpoint from the previous cascade stage.
#                           Becomes both the starting model AND the KL anchor.

set -euo pipefail

# ---- Per-skill knobs to fill in --------------------------------------------
SKILL_NAME="${SKILL_NAME:?set SKILL_NAME, e.g. appworld}"
SKILL_DATASETS=("${SKILL_DATASET:?set SKILL_DATASET}" "1.0")
SKILL_SYSTEM_PROMPT_FILE="${SKILL_SYSTEM_PROMPT_FILE:-}"  # leave empty to use per-row system prompts
SKILL_TOOLS="${SKILL_TOOLS:-generic_mcp}"                  # space-sep
SKILL_TOOL_CALL_NAMES="${SKILL_TOOL_CALL_NAMES:-mcp_tool}"
SKILL_TOOL_CONFIGS_JSON="${SKILL_TOOL_CONFIGS_JSON:-'{}'}"
SKILL_VERIFIER_REMAP="${SKILL_VERIFIER_REMAP:-}"          # e.g. general_rubric=rubric, empty if not needed
SKILL_MAX_STEPS="${SKILL_MAX_STEPS:-20}"
SKILL_RESPONSE_LENGTH="${SKILL_RESPONSE_LENGTH:-16384}"
SKILL_PACK_LENGTH="${SKILL_PACK_LENGTH:-18500}"
SKILL_VERIFICATION_REWARD="${SKILL_VERIFICATION_REWARD:-1.0}"
SKILL_BETA="${SKILL_BETA:-0.005}"  # KL anchor strength
SKILL_LR="${SKILL_LR:-1e-6}"
SKILL_TOTAL_EPISODES="${SKILL_TOTAL_EPISODES:-5120}"

# ---- Cascade plumbing ------------------------------------------------------
if [[ -z "${PREV_STAGE_CHECKPOINT:-}" ]]; then
    echo "ERROR: set PREV_STAGE_CHECKPOINT" >&2
    exit 1
fi

EXP_NAME="${EXP_NAME:-cascade_${SKILL_NAME}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

# Optional system-prompt-override args (only if SKILL_SYSTEM_PROMPT_FILE is set)
SYSTEM_PROMPT_ARGS=()
if [[ -n "${SKILL_SYSTEM_PROMPT_FILE}" ]]; then
    SYSTEM_PROMPT_ARGS=(--system_prompt_override_file "${SKILL_SYSTEM_PROMPT_FILE}")
fi

# Optional verifier remap
REMAP_ARGS=()
if [[ -n "${SKILL_VERIFIER_REMAP}" ]]; then
    REMAP_ARGS=(--remap_verifier "${SKILL_VERIFIER_REMAP}")
fi

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/general-tool-use \
    --priority "${PRIORITY:-urgent}" \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --budget ai2/oe-omai \
    --no_auto_dataset_cache \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    \
    `# --- cascade core ---` \
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
    --load_ref_policy True \
    --beta "${SKILL_BETA}" \
    \
    --dataset_mixer_list "${SKILL_DATASETS[@]}" \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length "${SKILL_RESPONSE_LENGTH}" \
    --pack_length "${SKILL_PACK_LENGTH}" \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --temperature 1.0 \
    --learning_rate "${SKILL_LR}" \
    --total_episodes "${SKILL_TOTAL_EPISODES}" \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --apply_verifiable_reward true \
    --verification_reward "${SKILL_VERIFICATION_REWARD}" \
    "${REMAP_ARGS[@]}" \
    --tool_parser_type vllm_qwen3_xml \
    --tools ${SKILL_TOOLS} \
    --tool_call_names ${SKILL_TOOL_CALL_NAMES} \
    --tool_configs ${SKILL_TOOL_CONFIGS_JSON} \
    --pool_size 128 \
    --max_steps "${SKILL_MAX_STEPS}" \
    "${SYSTEM_PROMPT_ARGS[@]}" \
    --backend_timeout 1800 \
    --active_sampling \
    --inflight_updates \
    --advantage_normalization_type centered \
    --kl_estimator 3 \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
    --keep_last_n_checkpoints -1 \
    --save_traces \
    --with_tracking \
    --seed 1 \
    --push_to_hub False
