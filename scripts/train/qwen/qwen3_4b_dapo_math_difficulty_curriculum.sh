#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_difficulty_curriculum}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-8}"
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
if [[ $# -gt 0 ]]; then
    shift
fi

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"

# Difficulty-annotated variant of hamishivi/DAPO-Math-17k-Processed_filtered
DATASET_WITH_DIFFICULTY="undfined/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples-ds"

TOTAL_EPISODES="${TOTAL_EPISODES:-128000}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-16}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-100}"

NUM_TRAINING_STEPS=$(( TOTAL_EPISODES / (NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT) ))

# Keep the easy bootstrap aligned with the first logging/eval window by default.
DIFFICULTY_CURRICULUM_EASY_FOCUS_STEPS="${DIFFICULTY_CURRICULUM_EASY_FOCUS_STEPS:-${LOCAL_EVAL_EVERY}}"
DIFFICULTY_CURRICULUM_WARMUP_STEPS="${DIFFICULTY_CURRICULUM_WARMUP_STEPS:-${DIFFICULTY_CURRICULUM_EASY_FOCUS_STEPS}}"
if (( NUM_TRAINING_STEPS <= DIFFICULTY_CURRICULUM_WARMUP_STEPS )); then
    DEFAULT_DIFFICULTY_CURRICULUM_TOTAL_STEPS=1
else
    DEFAULT_DIFFICULTY_CURRICULUM_TOTAL_STEPS=$(( NUM_TRAINING_STEPS - DIFFICULTY_CURRICULUM_WARMUP_STEPS ))
fi
DIFFICULTY_CURRICULUM_TOTAL_STEPS="${DIFFICULTY_CURRICULUM_TOTAL_STEPS:-${DEFAULT_DIFFICULTY_CURRICULUM_TOTAL_STEPS}}"

uv run python mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus $NUM_GPUS \
    --budget ai2/oe-adapt \
    -- \
uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_pass_at_k 32 \
    --eval_top_p 0.95 \
    --vllm_top_p 1.0 \
    --beta 0.0 \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout ${NUM_SAMPLES_PER_PROMPT_ROLLOUT} \
    --num_unique_prompts_rollout ${NUM_UNIQUE_PROMPTS_ROLLOUT} \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list "${DATASET_WITH_DIFFICULTY}" 1.0 \
    --dataset_mixer_list_splits "train" \
    --dataset_mixer_eval_list allenai/aime_2025_openinstruct 1.0 allenai/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits "train" \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path "Qwen/Qwen3-4B-Base" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes ${TOTAL_EPISODES} \
    --deepspeed_stage 2 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every ${LOCAL_EVAL_EVERY} \
    --save_freq ${SAVE_FREQ} \
    --checkpoint_state_freq ${CHECKPOINT_STATE_FREQ} \
    --gradient_checkpointing \
    --with_tracking \
    --send_slack_alerts \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --chat_template qwen_instruct_user_boxed_math \
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    --difficulty_curriculum_enabled true \
    --difficulty_curriculum_field difficulty \
    --difficulty_curriculum_easy_focus_steps ${DIFFICULTY_CURRICULUM_EASY_FOCUS_STEPS} \
    --difficulty_curriculum_bootstrap_target_bucket_ratio 0.125 \
    --difficulty_curriculum_warmup_target_bucket_ratio 0.5 \
    --difficulty_curriculum_final_target_bucket_ratio 1.0 \
    --difficulty_curriculum_warmup_steps ${DIFFICULTY_CURRICULUM_WARMUP_STEPS} \
    --difficulty_curriculum_total_steps ${DIFFICULTY_CURRICULUM_TOTAL_STEPS} \
    --difficulty_curriculum_min_hard_frac 0.05 \
    --difficulty_curriculum_max_hard_frac 0.50 \
    --difficulty_curriculum_bucket_sigma 0.0 \
    --difficulty_curriculum_easy_focus_sigma 0.0 \
    --difficulty_curriculum_uncertainty_weight 0.5 \
    --difficulty_curriculum_adaptive_enabled False \
    --difficulty_curriculum_strict_metadata true "$@"
