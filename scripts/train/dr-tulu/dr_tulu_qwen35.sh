#!/bin/bash
# DR-TULU training with Qwen 3.5 9B and evolving rubrics.
# Uses the dr-tulu-rl-data dataset with GPT-4.1 as rubric judge and generator.
#
# Launch via Beaker:
#   ./scripts/train/build_image_and_launch.sh scripts/train/dr-tulu/dr_tulu_qwen35.sh

EXP_NAME="${EXP_NAME:-dr_tulu_qwen35_4b}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="Qwen/Qwen3.5-4B"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS="rl-research/dr-tulu-rl-data 1.0"
DATASET_SPLITS="train"

PRIORITY="${PRIORITY:-high}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/dr-tulu-ablations \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env RUBRIC_JUDGE_MODEL=gpt-4.1 \
    --env RUBRIC_GENERATION_MODEL=gpt-4.1 \
    --secret SERPER_API_KEY=hamishivi_SERPER_API_KEY \
    --secret S2_API_KEY=hamishivi_S2_API_KEY \
    --secret JINA_API_KEY=hamishivi_JINA_API_KEY \
    --secret OPENAI_API_KEY=hamishivi_OPENAI_API_KEY \
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
    --total_episodes 1024000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    --tools serper_search jina_browse s2_search \
    --tool_call_names google_search browse_webpage snippet_search \
    --tool_configs '{}' '{}' '{"pool_size": 100}' \
    --pool_size 256 \
    --system_prompt_override_file scripts/train/dr-tulu/dr_tulu_adjusted.txt \
    --max_steps 10 \
    --backend_timeout 1800 \
    --save_traces \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --keep_last_n_checkpoints -1 \
    --kl_estimator 3 \
    --push_to_hub False
