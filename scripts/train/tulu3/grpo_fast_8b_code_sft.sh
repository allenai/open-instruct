base=SFT
description="test of https://github.com/allenai/open-instruct/pull/631"
exp_name=rlvr_tulu3.1_8b_${base}_grpo_fast_code
python mason.py \
    --cluster ai2/augusta \
    --image saurabhs/code_dev \
    --pure_docker_mode \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --preemptible \
    --num_nodes 4 \
    --description "${description}" \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --beta 0.01 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --try_launch_beaker_eval_jobs_on_weka \
    --kl_estimator 2 \
    --learning_rate 5e-7 \
    --dataset_mixer_list saurabh5/open-code-reasoning-rlvr 1.0 saurabh5/tulu-3-personas-code-rlvr 1.0 saurabh5/rlvr_acecoder 1.0 saurabh5/the-algorithm-python 1.0 saurabh5/llama-nemotron-rlvr 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list saurabh5/open-code-reasoning-rlvr 16 saurabh5/tulu-3-personas-code-rlvr 16 saurabh5/rlvr_acecoder 16 saurabh5/the-algorithm-python 16 saurabh5/llama-nemotron-rlvr 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision code_sft_allenai_Llama-3.1-Tulu-3-8B-SFT_n_1__8__1746057575 \
    --apply_verifiable_reward true \
    --code_api_url \$CODE_API_URL/test_program \
    --non_stop_penalty True \
    --oe_eval_tasks codex_humanevalplus:0-shot-chat-n5,mbppplus::openinstruct,cruxeval_input:pass@5,cruxeval_output:pass@5 \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --chat_template_name tulu_thinker_r1_style \
    --total_episodes 20000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_learners_per_node 6 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 10 \
    --lr_scheduler_type constant \
    --seed 1 \
    --local_eval_every 125 \
    --save_freq 200 \
    --gradient_checkpointing \
    --with_tracking


# --oe_eval_tasks gsm8k::tulu,bbh:cot-v1::tulu,codex_humanevalplus:0-shot-chat-n5,mbppplus::openinstruct,truthfulqa::tulu,cruxeval_input:pass@5,cruxeval_output:pass@5,ifeval::tulu \


#--model_name_or_path allenai/open_instruct_dev \
#--model_revision code_sft_allenai_Llama-3.1-Tulu-3-8B-DPO_n_1__8__1744778603 \

#--model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \

#--dataset_mixer_list saurabh5/open-code-reasoning-rlvr 1.0 saurabh5/tulu-3-personas-code-rlvr 10000 vwxyzjn/rlvr_acecoder 10000 vwxyzjn/the-algorithm-python 1.0 \


#--oe_eval_tasks gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct,drop::llama3,minerva_math::tulu,ifeval::tulu,popqa::tulu,mmlu:mc::tulu,mmlu:cot::summarize,alpaca_eval_v2::tulu,truthfulqa::tulu \
