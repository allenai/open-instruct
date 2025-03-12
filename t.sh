for dataset_combo in \
    "orz ai2-adapt-dev/rlvr_open_reasoner_math" \
; do
for learning_rate in 5e-7; do
for beta in 0.0; do
for nspp in 64; do
for kl_estimator in kl3; do
for num_unique_prompts_rollout in 128; do
read -r dataset_name dataset <<< "$dataset_combo"
exp_name="0306_force_eos_qwen2.5_7B_${dataset_name}_${RANDOM}"
echo $exp_name $dataset
python mason.py \
    --cluster ai2/augusta-google-1 \
    --image costah/open_instruct_dev_0311 --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --beta $beta \
    --num_samples_per_prompt_rollout $nspp \
    --num_unique_prompts_rollout $num_unique_prompts_rollout \
    --output_dir /weka/oe-adapt-default/costah/models/$exp_name \
    --kl_estimator $kl_estimator \
    --learning_rate $learning_rate \
    --dataset_mixer_list $dataset 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $dataset 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 8192 \
    --max_prompt_token_length 2048 \
    --response_length 6144 \
    --pack_length 16384 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stop_strings '"</answer>"' \
    --apply_r1_style_format_reward True \
    --apply_verifiable_reward True \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --chat_template_name r1_simple_chat_postpend_think \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
    --oe_eval_max_length 8192 \
    --temperature 1.0 \
    --total_episodes 10000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 8 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 20 \
    --lr_scheduler_type linear \
    --seed 1 \
    --num_evals 100 \
    --save_freq 3 \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
done
done
done
done
done
done



# https://wandb.ai/ai2-llm/open_instruct_internal/runs/96221yio/overview
exp_name="0213_qwen2.5_7B_math_lr_5e-7_kl3_0.0_16_half-m_4_${RANDOM}"
python mason.py \
    --cluster ai2/augusta-google-1 \
    --image costah/open_instruct_dev_0311 --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta 0.0 \
    --local_mini_batch_size 32 \
    --number_samples_per_prompt 16 \
    --output_dir /weka/oe-adapt-default/costah/models/$exp_name \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
    --save_freq 5 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka \
    --local_rollout_batch_size 4 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/math_ground_truth_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stop_strings '"</answer>"' \
    --add_r1_style_format_reward \
    --chat_template_name r1_simple_chat_postpend_think \
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
    --actor_num_gpus_per_node 8 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 4 \
    --lr_scheduler_type linear \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 200 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking