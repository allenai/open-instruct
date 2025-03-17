python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name tulu3.1_8b_grpo_fast \
    --beta 0.01 \
    --num_unique_prompts_rollout 64 \
    --num_samples_per_prompt_rollout 16 \
    --try_launch_beaker_eval_jobs_on_weka \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --apply_verifiable_reward True \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 2000000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_learners_per_node 6 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 100 \
    --save_freq 40 \
    --gradient_checkpointing \
    --gather_whole_model False \
    --with_tracking