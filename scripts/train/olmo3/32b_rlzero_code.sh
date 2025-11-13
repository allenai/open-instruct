#!/bin/bash

# OLMo 3 model
MODEL_NAME_OR_PATH="gs://ai2-llm/checkpoints/stego32-longcontext-run-3/step11921+11000+10000-nooptim-hf"
SHORT_MODEL_NAME="olmo3_32b_lc"

num_prompts_fn=6656
num_prompts_stdio=3328
DATASETS="saurabh5/rlvr_acecoder_filtered_filtered_olmo_completions_filtered ${num_prompts_fn} hamishivi/synthetic2-rlvr-code-compressed_filtered ${num_prompts_stdio} hamishivi/klear-code-rlvr_filtered ${num_prompts_stdio}"

#code evals
EVALS="codex_humanevalplus:0-shot-chat::tulu-thinker_RL0,mbppplus:0-shot-chat::tulu-thinker_RL0,livecodebench_codegeneration::tulu-thinker_RL0_no_think_tags_lite"
LOCAL_EVALS="saurabh5/rlvr_acecoder_filtered_filtered_olmo_completions_filtered 4 hamishivi/klear-code-rlvr_filtered 4"
LOCAL_EVAL_SPLITS="train train train train"

EXP_NAME="olmo3-32b_rlzero_code_${SHORT_MODEL_NAME}"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift  # Remove the image name from the argument list

cluster=ai2/augusta

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 12 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus 8 \
    --no_auto_dataset_cache \
    --budget ai2/oe-adapt \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --no_resampling_pass_rate 0.9 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --active_sampling \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_code_RL0 \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 12800 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 8 8 8 \
    --vllm_num_engines 16 \
    --gather_whole_model False \
    --vllm_tensor_parallel_size 4 \
    --inference_batch_size 160 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --output_dir /output/olmo3-32b-rlzero-code/checkpoints \
    --checkpoint_state_dir /output/olmo3-32b-rlzero-code/checkpoints \
    --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
    --code_max_execution_time 6.0 \
    --code_pass_rate_reward_threshold 0.99 \
    --gs_checkpoint_state_dir gs://ai2-llm/checkpoints/rlzero/olmo3-32b_rlzero-code/ \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --oe_eval_gpu_multiplier 4 \
    --oe_eval_max_length 32768 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --eval_priority high \
    --vllm_enforce_eager \
    --deepspeed_zpg 1 \
    --oe_eval_tasks $EVALS \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo3_auto $@ 