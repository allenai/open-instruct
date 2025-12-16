## Requires manual running
# first, launch the judge server:
beaker experiment create configs/judge_configs/general_verifier_judge.yaml \
    --name general_judge \
    --workspace ai2/tulu-3-results \
    --priority high

# then get the machine url. We set
JUDGE_BASE_URL=http://saturn-cs-aus-236.reviz.ai2.in:8000/v1

# finally, launch the training job:
python mason.py \
        --cluster ai2/jupiter --image hamishivi/open_instruct_judge_8 \
        --pure_docker_mode \
        --workspace ai2/tulu-thinker \
        --priority high \
        --preemptible \
        --num_nodes 2 \
        --max_retries 0 \
        --env HOSTED_VLLM_API_BASE=${JUDGE_BASE_URL} \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
            --exp_name 0906rl_judge_test_${RANDOM} \
            --dataset_mixer_list hamishivi/WebInstruct-verified-general-verifier-judge 1.0 \
            --dataset_mixer_list_splits train \
            --dataset_mixer_eval_list hamishivi/WebInstruct-verified-general-verifier-judge 16 \
            --dataset_mixer_eval_list_splits train \
            --max_token_length 10240 \
            --max_prompt_token_length 2048 \
            --response_length 8192 \
            --pack_length 16384 \
            --per_device_train_batch_size 1 \
            --num_unique_prompts_rollout 64 \
            --num_samples_per_prompt_rollout 16 \
            --model_name_or_path Qwen/Qwen2.5-7B \
            --stop_strings "</answer>" \
            --apply_verifiable_reward true \
            --apply_r1_style_format_reward true \
            --non_stop_penalty True \
            --non_stop_penalty_value 0.0 \
            --temperature 1.0 \
            --ground_truths_key ground_truth \
            --chat_template_name tulu_thinker_r1_style \
            --learning_rate 3e-7 \
            --total_episodes 200000 \
            --deepspeed_stage 2 \
            --num_epochs 1 \
            --num_learners_per_node 8 \
            --vllm_num_engines 8 \
            --vllm_tensor_parallel_size 1 \
            --beta 0.0 \
            --seed 3 \
            --local_eval_every 10 \
            --vllm_enforce_eager \
            --gradient_checkpointing \
            --push_to_hub false \
            --llm_judge_timeout 600 \
            --llm_judge_model "hosted_vllm/hamishivi/general-verifier" \
            --with_tracking
