#!/bin/bash
# Multi-task RL — Approach (3) cascade, STAGE 1: vanilla RL on dr_tulu.
#
# This is intentionally identical to scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh
# but with the relevant cascade knobs labeled. It's the starting point of the
# cascade and produces a checkpoint that stage 2 will consume.
#
# Launch:
#   ./scripts/train/build_image_and_launch.sh scripts/multi_task_rl/cascade_stage1_drtulu.sh
#
# After it finishes, note the final checkpoint path (printed in logs and at /output)
# and pass it as STAGE1_CHECKPOINT to cascade_stage2_swerl_with_kl_anchor.sh.

EXP_NAME="${EXP_NAME:-cascade_stage1_drtulu}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-4B}"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS="rl-research/dr-tulu-rl-data 0.05"
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
    \
    `# Stage 1: starting model is the pretrained base. No anchor needed yet.` \
    --beta 0.001 \
    --load_ref_policy True \
    \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --num_samples_per_prompt_rollout 32 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
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
    --total_episodes 5120 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    --tools serper_search jina_browse s2_search \
    --tool_call_names google_search browse_webpage snippet_search \
    --tool_configs '{}' '{}' '{}' \
    --pool_size 256 \
    --system_prompt_override_file scripts/train/dr-tulu/dr_tulu_adjusted.txt \
    --max_steps 12 \
    --backend_timeout 1800 \
    --save_traces \
    --seed 1 \
    --local_eval_every 100 \
    \
    `# Make sure the final checkpoint is saved — stage 2 needs it.` \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
    --keep_last_n_checkpoints -1 \
    \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --kl_estimator 3 \
    --push_to_hub False
