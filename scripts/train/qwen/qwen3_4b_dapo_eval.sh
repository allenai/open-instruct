#!/bin/bash

EXP_NAME="qwen3_4b_base_eval_dapo"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Base"
BEAKER_IMAGE="michaeln/open_instruct"

DATASETS="mnoukhov/dapo_math_14k_en_openinstruct 1.0"

LOCAL_EVALS="mnoukhov/dapo_math_14k_en_openinstruct 50"
LOCAL_EVAL_SPLITS="train"

# BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="michaeln/open_instruct"

CLUSTER="${CLUSTER:-ai2/saturn ai2/jupiter ai2/neptune}"
PRIORITY="${PRIORITY:-high}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus 2 \
    --budget ai2/oe-adapt \
    -- \
uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_only \
    --eval_pass_at_k 32 \
    --vllm_top_p 1.0 \
    --local_eval_every 100 \
    --beta 0.0 \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 32000 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False $@
