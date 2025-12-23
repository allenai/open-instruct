#!/bin/bash
# Note: This was originally a script that Saurabh came up to run some experiments.
# Finbarr has been using it a lot for testing, so we thought we'd check it in.
num_prompts=25376
exp_name=rlvr_ace_fn_and_og_ocr_stdio_from_base_with_perf_penalty
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
uv run python mason.py \
        --cluster ai2/jupiter \
        --image "$BEAKER_IMAGE" \
	--pure_docker_mode \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
	--preemptible \
        --num_nodes 2 \
	--description "Large (multi-node) test script." \
        --timeout 1h \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\&python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --load_ref_policy false \
        --num_samples_per_prompt_rollout 16 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 5e-7 \
        --per_device_train_batch_size 1 \
        --kl_estimator 2 \
        --dataset_mixer_list saurabh5/rlvr_acecoder_filtered ${num_prompts} saurabh5/open-code-reasoning-rlvr-stdio ${num_prompts} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list saurabh5/rlvr_acecoder_filtered 8 saurabh5/open-code-reasoning-rlvr-stdio 8 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 4096 \
        --pack_length 20480 \
        --model_name_or_path Qwen/Qwen2.5-7B \
        --chat_template_name tulu_thinker \
	--inflight_updates True \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --verbose False \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10_000 \
        --deepspeed_stage 2 \
        --num_learners_per_node 8 \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --code_api_url \$CODE_API_URL/test_program \
        --seed 1 \
        --local_eval_every 1 \
        --gradient_checkpointing \
        --try_launch_beaker_eval_jobs_on_weka True \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --oe_eval_max_length 32768 \
        --oe_eval_tasks "codex_humanevalplus:0-shot-chat-v1::tulu-thinker,mbppplus:0-shot-chat::tulu-thinker,livecodebench_codegeneration::tulu-thinker" \
		--checkpoint_state_freq 2 \
        --checkpoint_state_dir /tmp/checkpoint_test \
        --active_sampling \
        --async_steps 4 \
	--push_to_hub False
