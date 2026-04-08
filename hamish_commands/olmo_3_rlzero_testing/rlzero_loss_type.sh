#for masked_mean_denominator in token 16384 None group; do
for masked_mean_denominator in None; do
exp_name=olmo3_7b_rlzero_math_loss_denom_${masked_mean_denominator}
if [ "${masked_mean_denominator}" != "None" ]; then
    extra_args="--masked_mean_denominator ${masked_mean_denominator}"
else
    extra_args=""
fi
python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/augusta \
    --image hamishivi/open_instruct_dev_2611 \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority high \
    --preemptible \
    --num_nodes 8 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    --env HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --async_steps 8 \
        --inflight_updates \
        --no_resampling_pass_rate 0.875 \
        --truncated_importance_sampling_ratio_cap 2.0 \
        --advantage_normalization_type centered \
        --active_sampling \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --kl_estimator kl3 \
        --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list allenai/Dolci-RLZero-Math-7B 16 \
        --dataset_mixer_eval_list_splits train train \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 18432 \
        --model_name_or_path Qwen/Qwen3-8B-Base \
        --chat_template_name r1_simple_chat_postpend_think \
        --non_stop_penalty False \
        --temperature 1.0 \
        --total_episodes 512256 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 56 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 25 \
        --save_freq 100 \
        --checkpoint_state_freq 100 \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --clip_higher 0.272 \
        --oe_eval_max_length 32768 \
        --try_launch_beaker_eval_jobs_on_weka True \
        --eval_priority high \
        --eval_on_step_0 True \
        --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero \
        --oe_eval_gpu_multiplier 4 ${extra_args}
done
