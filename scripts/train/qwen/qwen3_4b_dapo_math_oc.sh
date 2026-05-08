#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-8}"
BEAKER_IMAGE="${BEAKER_IMAGE:-nathanl/open_instruct_auto}"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
if [[ "${1:-}" == "$BEAKER_USER"* ]]; then
    BEAKER_IMAGE="$1"
    shift
fi

CLUSTER="${CLUSTER:-ai2/jupiter ai2/ceres}"
PRIORITY="${PRIORITY:-urgent}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ai2/open-instruct-dev \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus $NUM_GPUS \
    --budget ai2/oe-other \
    -- \
uv run open_instruct/grpo.py \
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
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits "train" \
    --dataset_mixer_eval_list mnoukhov/aime_2025_openinstruct 1.0 mnoukhov/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits "train" \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path "Qwen/Qwen3-4B-Base" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
    --fsdp_shard_degree 4 \
    --fsdp_num_replicas 1 \
    --activation_memory_budget 0.5 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --send_slack_alerts \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --chat_template qwen_instruct_user_boxed_math \
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False "$@"
