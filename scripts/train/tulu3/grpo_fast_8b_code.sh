exp_name=rlvr_tulu3.1_8b_sft_grpo_fast_code
python mason.py \
    --cluster ai2/augusta-google-1 ai2/jupiter-cirrascale-2 \
    --image costah/open_instruct_dev_uv24 --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --description "open code reasoning on top of ocr sft model" \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup_code.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --beta 0.01 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --try_launch_beaker_eval_jobs_on_weka \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list saurabh5/open-code-reasoning-rlvr 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list saurabh5/open-code-reasoning-rlvr 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision code_sft_allenai_Llama-3.1-Tulu-3-8B-DPO_n_1__8__1744778603 \
    --apply_verifiable_reward false \
    --apply_code_reward true \
    --code_api_url \$CODE_API_URL/test_program \
    --non_stop_penalty True \
    --oe_eval_tasks gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct,drop::llama3,minerva_math::tulu,ifeval::tulu,popqa::tulu,mmlu:mc::tulu,mmlu:cot::summarize,alpaca_eval_v2::tulu,truthfulqa::tulu \
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
    --seed 1 \
    --num_evals 100 \
    --save_freq 40 \
    --gradient_checkpointing \
    --with_tracking 