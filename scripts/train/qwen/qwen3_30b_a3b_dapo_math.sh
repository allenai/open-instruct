#!/bin/bash
set -euo pipefail

# Full Qwen3-30B-A3B DAPO math baseline.
#
# This intentionally excludes MoE-specific stabilization:
#   - no router replay
#   - no router/load-balancing auxiliary loss
#   - no KL penalty
#   - no train/inference rho correction, clipping, or masking
#   - no truncated-completion masking
#
# Standard DAPO asymmetric policy clipping and dynamic sampling remain enabled.
#
# Topology:
#   - node 1: 8 DeepSpeed ZeRO-3 learner ranks
#   - node 2: 4 vLLM engines with tensor parallelism 2
#
# The default 128,000 episodes at 8 prompts x 16 samples runs 1,000
# optimizer steps.
#
# Run with:
#   ./scripts/train/build_image_and_launch.sh \
#       scripts/train/qwen/qwen3_30b_a3b_dapo_math.sh

EXP_NAME="${EXP_NAME:-qwen3_30b_a3b_base_dapo_baseline}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="Qwen/Qwen3-30B-A3B-Base"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
if (( $# > 0 )); then
    shift
fi

CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}"
PRIORITY="${PRIORITY:-high}"
MAX_RETRIES="${MAX_RETRIES:-5}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/${BEAKER_USER}/${RUN_NAME}}"

NUM_UNIQUE_PROMPTS=8
NUM_SAMPLES_PER_PROMPT=16
TOTAL_EPISODES=128000
NUM_TRAINING_STEPS=$((TOTAL_EPISODES / (NUM_UNIQUE_PROMPTS * NUM_SAMPLES_PER_PROMPT)))

if (( NUM_TRAINING_STEPS != 1000 )); then
    echo "Expected 1,000 training steps, got ${NUM_TRAINING_STEPS}" >&2
    exit 1
fi

uv run python mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster "${CLUSTER}" \
    --workspace "${WORKSPACE}" \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --max_retries "${MAX_RETRIES}" \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/aime_2025_openinstruct 1.0 allenai/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits train \
    --chat_template qwen_instruct_user_boxed_math \
    --apply_verifiable_reward True \
    --loss_fn dapo \
    --beta 0.0 \
    --load_ref_policy False \
    --use_vllm_logprobs False \
    --use_rho_correction False \
    --clip_higher 0.272 \
    --advantage_normalization_type centered \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT}" \
    --total_episodes "${TOTAL_EPISODES}" \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --non_stop_penalty False \
    --mask_truncated_completions False \
    --deepspeed_stage 3 \
    --deepspeed_zpg 8 \
    --deepspeed_offload_optimizer \
    --gather_whole_model False \
    --num_learners_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --eval_pass_at_k 32 \
    --eval_top_p 0.95 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --keep_last_n_checkpoints 1 \
    --save_final_model True \
    --try_auto_save_to_beaker False \
    --push_to_hub False \
    --seed 1 \
    --with_tracking "$@"
