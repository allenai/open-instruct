#!/bin/bash
# 1000-step version of scripts/train/debug/genac_smoke.sh, launched via mason.
#
# Same GenAC configuration as the smoke test, but using the VIP RL1K SAE + AP
# math rollout settings on Qwen3-4B as much as possible: 32 prompts, 8 samples
# per prompt, answer-prefix scalar value conditioning, SAE segmentation, and
# 1000 joint actor/critic RL steps after value warmup.
#
# GPU layout (16 GPUs on 2 nodes):
#   - 4x policy learners (DeepSpeed stage 3)
#   - 6x policy vLLM engines (TP=1)
#   - 5x generative-critic vLLM engines (TP=1)
#   - 1x GenValueTrainerActor (PyTorch, REINFORCE updates)
# (Rationale: keep learner world size fixed while spending the extra node on
# rollout and critic inference. Six policy engines can serve the 32x8 rollout
# batch without the sequential KV-cache waves seen with a single engine.)
#
# Step budget:
#   100 critic-only warmup steps (policy frozen via --value_warmup_steps)
# + 1000 joint actor/critic steps
# = 1100 total training steps
#
# total_episodes = 1100 * num_unique_prompts_rollout * num_samples_per_prompt_rollout
#                = 1100 * 32 * 8 = 281600
DDMM=$(date +"%d%m")
exp_name=genac_rl1k_${DDMM}_qwen3_4b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_RUN_ID=${exp_name}_$(date +%s) \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast_genvalue.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --async_steps 8 \
    --inflight_updates \
    --active_sampling \
    --no_resampling_pass_rate 0.875 \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 281600 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 6 \
    --vllm_tensor_parallel_size 1 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --gae_lambda 0.95 \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --use_sae \
    --sae_threshold 0.2 \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template answer_prefix \
    --use_generative_value_model \
    --gen_value_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --gen_value_vllm_num_engines 5 \
    --gen_value_vllm_tensor_parallel_size 1 \
    --gen_value_segmentation sae \
    --gen_value_max_segments 16 \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 1024 \
    --gen_value_conditioning gt \
    --gen_value_learning_rate 1e-6 \
    --gen_value_sync_freq 1
