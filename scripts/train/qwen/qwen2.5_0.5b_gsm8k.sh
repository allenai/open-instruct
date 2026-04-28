#!/bin/bash

EXP="${EXP:-}"
EXP_NAME="${EXP_NAME:-qwen2.5_0.5b_instruct_gsm8k_${EXP}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

DATASETS="${DATASETS:-mnoukhov/gsm8k-qwen2.5-0.5b-instruct-512samples-userprompt-quartiles 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt-quartiles 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test}"

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
shift || true

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
NUM_GPUS="${NUM_GPUS:-3}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --no_auto_dataset_cache \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus ${NUM_GPUS} \
    --budget ai2/oe-adapt \
    -- \
uv run open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --run_name ${RUN_NAME} \
    --beta 0.0 \
    --eval_top_p 1.0 \
    --async_steps 2 \
    --active_sampling \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 512 \
    --response_length 4096 \
    --pack_length 8192 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --chat_template_name qwen_instruct_user_boxed_math \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --num_learners_per_node 2 \
    --vllm_num_engines 1 \
    --clip_higher 0.28 \
    --load_ref_policy False \
    --with_tracking \
    --send_slack_alerts \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False "$@"
