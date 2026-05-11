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
WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}"
BUDGET="${BUDGET:-ai2/oe-omai}"

# Difficulty-annotated variant of hamishivi/DAPO-Math-17k-Processed_filtered
DATASET_WITH_DIFFICULTY="undfined/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples-ds"

TOTAL_EPISODES="${TOTAL_EPISODES:-128000}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-16}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-100}"

NUM_TRAINING_STEPS=$(( TOTAL_EPISODES / (NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT) ))

# Keep the bootstrap aligned with the first logging/eval window by default.
CURRICULUM_BOOTSTRAP_STEPS="${CURRICULUM_BOOTSTRAP_STEPS:-${LOCAL_EVAL_EVERY}}"
CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-${CURRICULUM_BOOTSTRAP_STEPS}}"
if (( NUM_TRAINING_STEPS <= CURRICULUM_WARMUP_STEPS )); then
    DEFAULT_CURRICULUM_TOTAL_STEPS=1
else
    DEFAULT_CURRICULUM_TOTAL_STEPS=$(( NUM_TRAINING_STEPS - CURRICULUM_WARMUP_STEPS ))
fi
CURRICULUM_TOTAL_STEPS="${CURRICULUM_TOTAL_STEPS:-${DEFAULT_CURRICULUM_TOTAL_STEPS}}"
CURRICULUM_METADATA_FIELD="${CURRICULUM_METADATA_FIELD:-difficulty}"
CURRICULUM_POSTERIOR_MEAN_FIELD="${CURRICULUM_POSTERIOR_MEAN_FIELD:-posterior_mean}"
CURRICULUM_BUCKET_INDEX_FIELD="${CURRICULUM_BUCKET_INDEX_FIELD:-bucket_index}"
CURRICULUM_BUCKET_COUNT_FIELD="${CURRICULUM_BUCKET_COUNT_FIELD:-bucket_count}"
CURRICULUM_QUANTILE_FIELD="${CURRICULUM_QUANTILE_FIELD:-expected_quantile}"
CURRICULUM_STRICT_METADATA="${CURRICULUM_STRICT_METADATA:-true}"
CURRICULUM_BOOTSTRAP_TARGET="${CURRICULUM_BOOTSTRAP_TARGET:-0.125}"
CURRICULUM_WARMUP_TARGET="${CURRICULUM_WARMUP_TARGET:-0.5}"
CURRICULUM_FINAL_TARGET="${CURRICULUM_FINAL_TARGET:-1.0}"
CURRICULUM_MIN_HARD_FRAC="${CURRICULUM_MIN_HARD_FRAC:-0.05}"
CURRICULUM_MAX_HARD_FRAC="${CURRICULUM_MAX_HARD_FRAC:-0.50}"
CURRICULUM_BUCKET_SIGMA="${CURRICULUM_BUCKET_SIGMA:-0.0}"
CURRICULUM_BOOTSTRAP_SIGMA="${CURRICULUM_BOOTSTRAP_SIGMA:-0.0}"
CURRICULUM_UNCERTAINTY_WEIGHT="${CURRICULUM_UNCERTAINTY_WEIGHT:-0.5}"
CURRICULUM_ADAPTIVE="${CURRICULUM_ADAPTIVE:-false}"
CURRICULUM_ADAPTIVE_UPDATE_EVERY="${CURRICULUM_ADAPTIVE_UPDATE_EVERY:-50}"
CURRICULUM_ADAPTIVE_LEARNING_WEIGHT="${CURRICULUM_ADAPTIVE_LEARNING_WEIGHT:-0.7}"
CURRICULUM_ADAPTIVE_EXPLORATION_WEIGHT="${CURRICULUM_ADAPTIVE_EXPLORATION_WEIGHT:-0.3}"
CURRICULUM_ADAPTIVE_BLEND="${CURRICULUM_ADAPTIVE_BLEND:-0.5}"
CURRICULUM_MIN_QUANTILE="${CURRICULUM_MIN_QUANTILE:-0.0}"
CURRICULUM_MAX_QUANTILE="${CURRICULUM_MAX_QUANTILE:-1.0}"

CURRICULUM_ARGS=(
    --curriculum difficulty
    --curriculum_metadata_field "${CURRICULUM_METADATA_FIELD}"
    --curriculum_posterior_mean_field "${CURRICULUM_POSTERIOR_MEAN_FIELD}"
    --curriculum_bucket_index_field "${CURRICULUM_BUCKET_INDEX_FIELD}"
    --curriculum_bucket_count_field "${CURRICULUM_BUCKET_COUNT_FIELD}"
    --curriculum_quantile_field "${CURRICULUM_QUANTILE_FIELD}"
    --curriculum_bootstrap_steps "${CURRICULUM_BOOTSTRAP_STEPS}"
    --curriculum_bootstrap_target "${CURRICULUM_BOOTSTRAP_TARGET}"
    --curriculum_warmup_target "${CURRICULUM_WARMUP_TARGET}"
    --curriculum_final_target "${CURRICULUM_FINAL_TARGET}"
    --curriculum_warmup_steps "${CURRICULUM_WARMUP_STEPS}"
    --curriculum_total_steps "${CURRICULUM_TOTAL_STEPS}"
    --curriculum_min_hard_frac "${CURRICULUM_MIN_HARD_FRAC}"
    --curriculum_max_hard_frac "${CURRICULUM_MAX_HARD_FRAC}"
    --curriculum_bucket_sigma "${CURRICULUM_BUCKET_SIGMA}"
    --curriculum_bootstrap_sigma "${CURRICULUM_BOOTSTRAP_SIGMA}"
    --curriculum_uncertainty_weight "${CURRICULUM_UNCERTAINTY_WEIGHT}"
    --curriculum_adaptive "${CURRICULUM_ADAPTIVE}"
    --curriculum_adaptive_update_every "${CURRICULUM_ADAPTIVE_UPDATE_EVERY}"
    --curriculum_adaptive_learning_weight "${CURRICULUM_ADAPTIVE_LEARNING_WEIGHT}"
    --curriculum_adaptive_exploration_weight "${CURRICULUM_ADAPTIVE_EXPLORATION_WEIGHT}"
    --curriculum_adaptive_blend "${CURRICULUM_ADAPTIVE_BLEND}"
    --curriculum_min_quantile "${CURRICULUM_MIN_QUANTILE}"
    --curriculum_max_quantile "${CURRICULUM_MAX_QUANTILE}"
    --curriculum_strict_metadata "${CURRICULUM_STRICT_METADATA}"
)

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
    --budget ${BUDGET} \
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
    "${CURRICULUM_ARGS[@]}" "$@"
