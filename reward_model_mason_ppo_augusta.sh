# make sure to match up the GPUs. E.g.,
# `--actor_num_gpus_per_node 6 8`
# `--vllm_tensor_parallel_size 2`
# translates to 6 + 2 + 8 = 16 GPUs
# which matches up with `--num_nodes 2 --gpus 8`
revision=reward_modeling__1__1739943850
for beta in 0.05; do
exp_name="0319_ppo_sftbase_${beta}_${revision}"
python mason.py \
    --cluster ai2/augusta-google-1 \
    --workspace ai2/reward-bench-v2 \
    --priority high \
    --preemptible \
    --image saumyam/open-instruct-0318-nonrlvr --pure_docker_mode \
    --budget ai2/oe-adapt \
    --num_nodes 2 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_gcs.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /output \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list saumyamalik/tulu-3-8b-preference-mixture-with-messages 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list saumyamalik/tulu-3-8b-preference-mixture-with-messages 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision $revision\
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key chosen \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 4 \
    --local_rollout_batch_size 4 \
    --actor_num_gpus_per_node 6 8 \
    --vllm_tensor_parallel_size 2 \
    --vllm_enforce_eager \
    --apply_verifiable_reward false \
    --eval_priority normal \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 100 \
    --reward_model_multiplier 1.0 \
    --gradient_checkpointing \
    --wandb_project_name reward-models \
    --with_tracking \
    --gs_bucket_path gs://ai2-llm/post-training/
done