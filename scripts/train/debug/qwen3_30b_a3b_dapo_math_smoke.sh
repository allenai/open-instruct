#!/bin/bash
set -eo pipefail

# Three-step production-model smoke test for Qwen3-MoE live weight sync.
#
# Topology:
#   - node 1: 8 DeepSpeed ZeRO-3 learner ranks
#   - node 2 on CUDA 12: 4 vLLM engines with tensor parallelism 2
#   - node 2 on CUDA 13/B300: 8 vLLM engines with tensor parallelism 1
#
# Run with:
#   ./scripts/train/build_image_and_launch.sh \
#       scripts/train/debug/qwen3_30b_a3b_dapo_math_smoke.sh
#
# The full 1,000-step baseline is:
#   scripts/train/qwen/qwen3_30b_a3b_dapo_math.sh

EXP_NAME="${EXP_NAME:-qwen3_30b_a3b_dapo_math_smoke}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="Qwen/Qwen3-30B-A3B-Base"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../qwen/qwen3_30b_a3b_dapo_math_profile.sh"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
CUDA_VERSION=$(qwen3_30b_a3b_cuda_version_for_image "${BEAKER_IMAGE}")
IFS="|" read -r DEFAULT_CLUSTER DEFAULT_VLLM_NUM_ENGINES DEFAULT_VLLM_TENSOR_PARALLEL_SIZE \
    DEFAULT_DEEPSPEED_OFFLOAD_OPTIMIZER <<< "$(qwen3_30b_a3b_hardware_profile "${CUDA_VERSION}")"

CLUSTER="${CLUSTER:-${DEFAULT_CLUSTER}}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-${DEFAULT_VLLM_NUM_ENGINES}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${DEFAULT_VLLM_TENSOR_PARALLEL_SIZE}}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-${DEFAULT_DEEPSPEED_OFFLOAD_OPTIMIZER}}"
PRIORITY="${PRIORITY:-urgent}"

if [[ "${CUDA_VERSION}" == "13" && "${CLUSTER}" != "ai2/holmes" ]]; then
    echo "CUDA 13 Qwen3-30B-A3B runs require CLUSTER=ai2/holmes; got ${CLUSTER}." >&2
    exit 1
fi
if (( VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE != 8 )); then
    echo "The vLLM topology must use exactly 8 inference GPUs; got " \
        "${VLLM_NUM_ENGINES} engines x TP=${VLLM_TENSOR_PARALLEL_SIZE}." >&2
    exit 1
fi
if [[ "${DEEPSPEED_OFFLOAD_OPTIMIZER}" != "true" && "${DEEPSPEED_OFFLOAD_OPTIMIZER}" != "false" ]]; then
    echo "DEEPSPEED_OFFLOAD_OPTIMIZER must be true or false." >&2
    exit 1
fi

DEEPSPEED_OFFLOAD_OPTIMIZER_ARG=""
if [[ "${DEEPSPEED_OFFLOAD_OPTIMIZER}" == "true" ]]; then
    DEEPSPEED_OFFLOAD_OPTIMIZER_ARG="--deepspeed_offload_optimizer"
fi

# Four prompts x four samples = sixteen episodes per optimizer step.
# Forty-eight total episodes therefore runs exactly three optimizer steps,
# which forces both the initial sync and a post-update sync before exit.
NUM_UNIQUE_PROMPTS=4
NUM_SAMPLES_PER_PROMPT=4
TOTAL_EPISODES=$((3 * NUM_UNIQUE_PROMPTS * NUM_SAMPLES_PER_PROMPT))

# mason.py injects checkpoint_state_dir, which requires a positive frequency.
# A frequency of 100 is beyond this three-step run, so no state is written.
uv run python mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster "${CLUSTER}" \
    --workspace ai2/open-instruct-dev \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --max_retries 0 \
    --timeout 2h \
    --artifact_ttl 1d \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --chat_template qwen_instruct_user_boxed_math \
    --apply_verifiable_reward true \
    --loss_fn dapo \
    --beta 0.0 \
    --clip_higher 0.272 \
    --advantage_normalization_type centered \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT}" \
    --total_episodes "${TOTAL_EPISODES}" \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --non_stop_penalty False \
    --mask_truncated_completions False \
    --deepspeed_stage 3 \
    --deepspeed_zpg 8 \
    ${DEEPSPEED_OFFLOAD_OPTIMIZER_ARG:+"${DEEPSPEED_OFFLOAD_OPTIMIZER_ARG}"} \
    --gather_whole_model False \
    --num_learners_per_node 8 \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --load_ref_policy False \
    --local_eval_every -1 \
    --save_freq -1 \
    --checkpoint_state_freq 100 \
    --try_auto_save_to_beaker False \
    --push_to_hub False \
    --seed 1 \
    --with_tracking
