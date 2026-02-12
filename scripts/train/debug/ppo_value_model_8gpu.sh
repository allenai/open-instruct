#!/bin/bash
# VAPO (Value-model-based Augmented PPO) test script
# Matches rlzero_seqlen.sh configuration but with VAPO value model
# Uses 4 nodes (2 learner nodes + 2 vLLM nodes), 16k seqlen, no sequence parallelism
# Reference: https://arxiv.org/abs/2504.05118

DDMM=$(date +"%d%m")
exp_name=vapo_${DDMM}_7b_rlzero_math_16k
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
    --kl_estimator 2 \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/Dolci-RLZero-Math-7B 16 \
    --dataset_mixer_eval_list_splits train train \
    --max_prompt_token_length 2048 \
    --response_length 14336 \
    --pack_length 16384 \
    --model_name_or_path allenai/Olmo-3-1025-7B \
    --chat_template_name olmo_thinker_rlzero \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512256 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 8 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 16 \
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
    --oe_eval_max_length 16384 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --eval_priority normal \
    --eval_on_step_0 True \
    --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero \
    --oe_eval_gpu_multiplier 4 \
    --use_value_model \
    --value_loss_coef 0.5 \
    --value_learning_rate 2e-6 \
    --vf_clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --value_warmup_steps 150 \
    --reset_optimizer_after_value_warmup \
    --decoupled_gae \
    --length_adaptive_gae \
    --length_adaptive_gae_alpha 0.05 \
    --loss_fn dapo \
    --clip_higher 0.28 \
    --push_to_hub False
