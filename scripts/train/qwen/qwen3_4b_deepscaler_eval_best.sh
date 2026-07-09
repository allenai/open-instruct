#!/bin/bash

# Eval-only run (open_instruct/grpo.py --eval_only) of a single HF checkpoint on the
# held-out competition sets (BRUMO 2025, HMMT Feb/Nov 2025, AIME 2025), matching the
# in-training AIME eval methodology of scripts/train/qwen/qwen3_4b_deepscaler_math.sh
# (pass@32, temperature 1.0, eval_top_p 0.95, 8192-token responses,
# qwen_instruct_user_boxed_math chat template).
#
# Required env vars:
#   BEAKER_IMAGE        image containing the --eval_only port
#   MODEL_NAME_OR_PATH  HF-format checkpoint dir (e.g. on weka) or HF hub name
# Optional:
#   EXP_NAME, BEST_STEP (training step the metrics are attributed to), WANDB_GROUP_NAME

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?set MODEL_NAME_OR_PATH to the HF checkpoint}"
BEAKER_IMAGE="${BEAKER_IMAGE:?set BEAKER_IMAGE}"
EXP_NAME="${EXP_NAME:-qwen3_4b_deepscaler_eval_best}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
BEST_STEP="${BEST_STEP:-1}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-deepscaler_eval_best}"

NUM_GPUS="${NUM_GPUS:-4}"
CLUSTER="${CLUSTER:-ai2/jupiter ai2/saturn ai2/neptune}"
PRIORITY="${PRIORITY:-high}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/brumo_2025_openinstruct 1.0 mnoukhov/hmmt_feb_2025_openinstruct 1.0 mnoukhov/hmmt_nov_2025_openinstruct 1.0 mnoukhov/aime_2025_openinstruct 1.0}"

uv run mason.py \
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
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --gpus $NUM_GPUS \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_only \
    --eval_only_set_checkpoint ${BEST_STEP} \
    --eval_pass_at_k 32 \
    --eval_top_p 0.95 \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --dataset_mixer_list mnoukhov/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_list_splits "train" \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits "train" \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --eval_response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --chat_template qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --apply_verifiable_reward true \
    --beta 0.0 \
    --load_ref_policy False \
    --async_steps 1 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --total_episodes 128 \
    --vllm_num_engines $NUM_GPUS \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching \
    --seed 1 \
    --with_tracking \
    --wandb_group_name "${WANDB_GROUP_NAME}" \
    --push_to_hub False "$@"
