#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen2.5_0.5b_instruct_gsm8k}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

DATASETS="${DATASETS:-ai2-adapt-dev/rlvr_gsm8k_zs 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test}"

BEAKER_IMAGE="michaeln/open_instruct"

CLUSTER="${CLUSTER:-ai2/neptune ai2/jupiter ai2/ceres ai2/titan}"
PRIORITY="${PRIORITY:-high}"
NUM_GPUS="${NUM_GPUS:-4}"

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
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus ${NUM_GPUS} \
    --budget ai2/oe-adapt \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run --active open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --run_name $EXP_NAME \
    --beta 0.0 \
    --eval_pass_at_k 4 \
    --eval_top_p 0.95 \
    --vllm_top_p 1.0 \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 512 \
    --response_length 4096 \
    --pack_length 4608 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --num_learners_per_node 2 \
    --vllm_tensor_parallel_size 2 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --with_tracking \
    --push_to_hub False $@

    # --checkpoint_state_freq 200 \
    # --keep_last_n_checkpoints -1 \

    # --eval_priority normal \
    # --try_launch_beaker_eval_jobs_on_weka True \
    # --oe_eval_max_length 32768 \
    # --oe_eval_gpu_multiplier 2  \
    # --oe_eval_beaker_image michaeln/oe_eval_internal \
    # --oe_eval_tasks $EVALS \
