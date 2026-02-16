#!/bin/bash

EXP_NAME="qwen3_4b_it_deepmath"
MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
DATASETS="mnoukhov/DeepMath-103K-openinstruct 1.0"

LOCAL_EVALS="allenai/aime2024-25-rlvr 32"
LOCAL_EVAL_SPLITS="test_2024 test_2024"

EVALS="aime:2024::justrl,aime:2025::justrl"

# BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="michaeln/open_instruct"
cluster=ai2/titan

# Check if the first argument starts with the value of $BEAKER_NAME
# if [[ "$1" == "$BEAKER_USER"* ]]; then
#     BEAKER_IMAGE="$1"
#     shift
# fi

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/scaling-rl \
    --priority low \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus 4 \
    --budget ai2/oe-adapt \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --no_resampling_pass_rate 0.875 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 24000 \
    --pack_length 32000 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --user_prompt_transform "{prompt}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}." \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 3 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --oe_eval_max_length 32768 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --eval_priority normal \
    --oe_eval_tasks $EVALS \
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --oe_eval_gpu_multiplier 2  \
    --oe_eval_beaker_image michaeln/oe_eval_internal \
    --push_to_hub False $@
