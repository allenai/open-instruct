exp_name="grpo_fast_code_${RANDOM}"
python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --output_dir /weka/oe-adapt-default/saurabhs/models/$exp_name \
    --dataset_mixer_list vwxyzjn/rlvr_acecoder 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list vwxyzjn/rlvr_acecoder 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path HuggingFaceTB/SmolLM2-135M \
    --stop_strings "eos" \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --learning_rate 3e-7 \
    --total_episodes 87040 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --num_evals 20 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.2 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --with_tracking


    # per_device_batch_size needs to be 1!! 
