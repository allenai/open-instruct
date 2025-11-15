#!/bin/bash

MODEL_NAME_OR_PATH="/weka/oe-adapt-default/michaeln/checkpoints/olmo3-7b-base"

python mason.py \
    --task_name grpo_debug_olmo3 \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image michaeln/open_instruct_rlzero \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- \
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 4096 \
    --pack_length 4608 \
    --async_steps 2 \
    --inflight_updates \
    --active_fill_completions \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_dapo \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --learning_rate 3e-7 \
    --total_episodes 1600 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --beta 0. \
    --seed 3 \
    --local_eval_every 100 \
    --gradient_checkpointing \
    --push_to_hub false
