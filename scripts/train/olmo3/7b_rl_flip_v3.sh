#!/bin/bash

# OLMo 3 model
# MODEL_NAME_OR_PATH="allenai/Olmo-3-7B-Instruct-DPO"
MODEL_NAME_OR_PATH="/weka/oe-adapt-default/yikew/olmo/Olmo-3-7B-Instruct-DPO"

DATASETS="yikeee/rlvr_general_chat_flip 1.0"
# DATASETS="allenai/Dolci-RLZero-Math-7B 1.0"

# LOCAL_EVALS="hamishivi/rlvr_general_mix 8"
LOCAL_EVALS="yikeee/rlvr_general_chat_flip 8"
LOCAL_EVAL_SPLITS="train"

EVALS="alpaca_eval_v3::hamish_zs_reasoning_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek"

JUDGE_BASE_URL=http://saturn-cs-aus-252.reviz.ai2.in:8001/v1

EXP_NAME="grpo_general_flip_instruct"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="yikew/open_instruct_olmo4"
shift

cluster=ai2/titan

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/tulu-thinker \
    --priority high \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 3 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env WANDB_ENTITY=ai2-llm \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --env VLLM_RPC_TIMEOUT=60000 \
    --gpus 8 \
    --env HOSTED_VLLM_API_BASE=${JUDGE_BASE_URL} \
    --budget ai2/oe-adapt \
    -- \
source configs/beaker_configs/ray_node_setup.sh \&\& \
python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 1 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 24 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo123 \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 10000000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --backend_timeout 3600 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --llm_judge_model hosted_vllm/Qwen/Qwen3-1.7B \
    --llm_judge_timeout 3600 \
    --llm_judge_max_tokens 2048 \
    --llm_judge_max_context_length 20000 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
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
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_auto \
    --output_dir /output/olmo3-7b-rlzero-general/checkpoints $@
