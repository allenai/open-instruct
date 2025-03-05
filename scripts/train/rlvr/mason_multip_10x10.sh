# for dataset_combo in \
#     "orz ai2-adapt-dev/rlvr_open_reasoner_math" \
# ; do
# for learning_rate in 5e-7; do
# for beta in 0.0; do
# for nspp in 64; do
# for m in half-m m; do
# for kl_estimator in kl3; do
# for local_rollout_batch_size in 4; do
# if [ $m == "half-m" ]; then
#     local_mini_batch_size=$(($local_rollout_batch_size * $nspp / 2))
# else
#     local_mini_batch_size=$(($local_rollout_batch_size * $nspp))
# fi
# read -r dataset_name dataset <<< "$dataset_combo"
# echo --- local_mini_batch_size=$local_mini_batch_size
# echo --- num_gradient_updates=$(($local_rollout_batch_size * $nspp / $local_mini_batch_size))
# exp_name="0226_qwen2.5_7B_${dataset_name}_${m}_${RANDOM}"
# echo $exp_name $dataset
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 --image costah/open_instruct_dev_0224 --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name multiplication-RL \
    --beta 0.01 \
    --local_mini_batch_size 64 \
    --number_samples_per_prompt 8 \
    --output_dir /output \
    --local_rollout_batch_size 64 \
    --kl_estimator kl3 \
    --learning_rate 3e-7 \
    --dataset_mixer_list nouhad/multiplication_train_1000_10x10.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list nouhad/multiplication_test_100_12x12 1.0 nouhad/multiplication_test_100_6x6 1.0 nouhad/multiplication_test_100_2x2 1.0\
    --dataset_mixer_eval_list_splits train \
    --max_token_length 8192 \
    --max_prompt_token_length 200 \
    --response_length 3500 \
    --model_name_or_path Qwen/Qwen2.5-Math-7B \
    --stop_strings '"</answer>"' \
    --add_r1_style_format_reward \
    --chat_template_name r1_simple_chat_postpend_think \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
    --non_stop_penalty False \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 10000000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --actor_num_gpus_per_node 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 4 \
    --enable_prefix_caching \
    --lr_scheduler_type linear \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 100 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka  \
    --gradient_checkpointing \
    --with_tracking