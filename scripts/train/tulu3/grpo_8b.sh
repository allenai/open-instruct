for learning_rate in 5e-7; do
for beta in 0.01; do
for nspp in 16; do
for m in half-m ; do
for kl_estimator in kl3; do
local_rollout_batch_size=4
if [ $m == "half-m" ]; then
    local_mini_batch_size=$(($local_rollout_batch_size * $nspp / 2))
else
    local_mini_batch_size=$(($local_rollout_batch_size * $nspp))
fi
exp_name="0204_lr_scan_grpo_math_lr_${learning_rate}_${kl_estimator}_${beta}_${nspp}_${m}_${RANDOM}"
full_bsz=$(($local_rollout_batch_size * nspp * (7) * 2))
echo $exp_name:
echo --- local_mini_batch_size=$local_mini_batch_size
echo --- full_bsz=$full_bsz
echo --- num_gradient_updates=$(($local_rollout_batch_size * $nspp / $local_mini_batch_size))
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --max_retries 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --local_mini_batch_size $local_mini_batch_size \
    --number_samples_per_prompt $nspp \
    --output_dir /weka/oe-adapt-default/costah/models/$exp_name \
    --local_rollout_batch_size $local_rollout_batch_size \
    --kl_estimator $kl_estimator \
    --learning_rate $learning_rate \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --total_episodes 10000000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --actor_num_gpus_per_node 4 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 30 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
done
done
done
done
done
