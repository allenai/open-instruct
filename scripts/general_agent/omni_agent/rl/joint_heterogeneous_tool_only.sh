#!/bin/bash
# Multi-task RL — Approach (2): single run, heterogeneous batches, function-calling tools only.
#
# This runs TODAY without code changes. It assumes you have pre-built a single HF
# dataset (or two-three) where each row carries:
#   - messages: skill-specific system prompt baked in as messages[0]
#   - ground_truth: per-skill ground truth (str / list / JSON for rubric)
#   - dataset: name of the verifier that should score this row
#              (e.g. "math", "code", "general_rubric")
#   - tools: list of tool call names this row may use (e.g. ["google_search", "browse_webpage"])
#   - env_config: per-row env kwargs (often {"env_configs": []} for tool-only skills)
#
# See docs/algorithms/multi_task_rl.md §5 for the dataset schema and §3 for caveats.
#
# Launch:
#   ./scripts/train/build_image_and_launch.sh scripts/multi_task_rl/joint_heterogeneous_tool_only.sh
#
# This example mixes:
#   - dr-tulu-rl-data (research / citation skill, evolving rubrics)
#   - <user-provided MCP dataset> (any function-calling MCP skill)
#   - <user-provided math dataset> (verified scalar reward)
#
# REPLACE the dataset names below with your unified datasets.

EXP_NAME="${EXP_NAME:-multi_task_joint_hetero_tools}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-4B}"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

# Mix three skills. Use 1.0 = 100% of each, or "N" for absolute counts.
# IMPORTANT: every dataset listed here must already have the per-row columns described above.
DATASETS=(
    "rl-research/dr-tulu-rl-data" "0.05"
    # "<your-org>/<your-mcp-rl-dataset>" "1.0"
    # "<your-org>/<your-math-rl-dataset>" "0.5"
)
DATASET_SPLITS="train"

PRIORITY="${PRIORITY:-urgent}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/general-tool-use \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --budget ai2/oe-omai \
    --no_auto_dataset_cache \
    --env RUBRIC_JUDGE_MODEL=gpt-4.1 \
    --env RUBRIC_GENERATION_MODEL=gpt-4.1 \
    --secret SERPER_API_KEY=shashankg_SERPER_API_KEY \
    --secret S2_API_KEY=shashankg_S2_API_KEY \
    --secret JINA_API_KEY=shashankg_JINA_API_KEY \
    --secret OPENAI_API_KEY=shashankg_OPENAI_API_KEY \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --beta 0.001 \
    --load_ref_policy True \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list "${DATASETS[@]}" \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18500 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --non_stop_penalty False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 10240 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    \
    `# Register every tool that ANY skill in the dataset might call.` \
    `# Per-row 'tools' column filters which tools each prompt actually sees.` \
    --tools serper_search jina_browse s2_search generic_mcp \
    --tool_call_names google_search browse_webpage snippet_search mcp_tool \
    --tool_configs '{}' '{}' '{}' '{"host": "REPLACE_WITH_YOUR_MCP_HOST", "port": 8765, "transport_type": "http"}' \
    \
    --pool_size 256 \
    \
    `# NO --system_prompt_override_file: each row carries its own system prompt in messages[0].` \
    `# Bake skill-specific system prompts into the dataset before running.` \
    \
    --max_steps 12 \
    --backend_timeout 1800 \
    --save_traces \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 200 \
    --checkpoint_state_freq 200 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --keep_last_n_checkpoints -1 \
    --kl_estimator 3 \
    \
    `# Centered advantage normalization stays per-prompt-group (= per-skill).` \
    --advantage_normalization_type centered \
    --push_to_hub False
