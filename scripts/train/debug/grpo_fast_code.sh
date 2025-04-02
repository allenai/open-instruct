exp_name="tulu3_8b_grpo_fast_code_${RANDOM}"
CODE_API_URL=https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod
python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --beta 0.01 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --try_launch_beaker_eval_jobs_on_weka \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list vwxyzjn/rlvr_acecoder 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list vwxyzjn/rlvr_acecoder 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --apply_verifiable_reward false \
    --apply_ace_coder_reward true \
    --ace_coder_api_url $CODE_API_URL/test_program \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 2000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_learners_per_node 6 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 10 \
    --lr_scheduler_type constant \
    --seed 66 \
    --num_evals 100 \
    --save_freq 40 \
    --gradient_checkpointing
    # --with_tracking