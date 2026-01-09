#!/bin/bash

MODEL_NAME_OR_PATH="allenai/Olmo-3-1025-7B"
DATASETS="allenai/Dolci-RLZero-IF-7B 1.0"

LOCAL_EVALS="allenai/Dolci-RLZero-IF-7B 8"
LOCAL_EVAL_SPLITS="train"

EVALS="ifeval::hamish_zs_reasoning_deepseek"

EXP_NAME="olmo3_7b_rlzero_if"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift

cluster=ai2/augusta

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/olmo-instruct \
    --priority high \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 5 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_rlzero \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 10000000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 32 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 50 \
    --checkpoint_state_freq 200 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --keep_last_n_checkpoints -1 \
    --mask_truncated_completions True \
    --oe_eval_max_length 16384 \
    --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks $EVALS \
    --eval_on_step_0 True \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
    --output_dir /output/olmo3-7b-rlzero-if/checkpoints $@
