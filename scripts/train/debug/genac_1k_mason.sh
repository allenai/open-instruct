#!/bin/bash
# 1000-step version of scripts/train/debug/genac_smoke.sh, launched via mason.
#
# Same GenAC configuration as the smoke test (conditioning=none, no scalar
# value head, SAE segmentation, sync every step) but scaled up enough to see
# real training dynamics on Qwen3-4B over ~1000 policy steps. This is the
# script to run when you want to watch gen_value/mse actually trend down and
# gen_value/v_hat_mean converge toward gen_value/outcome_mean.
#
# GPU layout (8 GPUs on 1 node):
#   - 4x policy learners (DeepSpeed stage 3)
#   - 1x policy vLLM engine (TP=1)
#   - 2x generative-critic vLLM engines (TP=1)
#   - 1x GenValueTrainerActor (PyTorch, REINFORCE updates)
# (Rationale: with num_samples_per_prompt_rollout=16 the gen-value pool has to
# score 16x more responses per step than the policy generates per prompt, so
# the gen-value vLLM pool is the bottleneck. Give it more engines than policy.)
#
# Step budget:
#   100 critic-only warmup steps (policy frozen via --value_warmup_steps)
# + 1000 joint actor/critic steps
# = 1100 total training steps
#
# total_episodes = 1100 * num_unique_prompts_rollout * num_samples_per_prompt_rollout
#                = 1100 * 16 * 16 = 281600
DDMM=$(date +"%d%m")
exp_name=genac_1k_${DDMM}_qwen3_4b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_RUN_ID=${exp_name}_$(date +%s) \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast_genvalue.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8096 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 281600 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_top_p 1.0 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 250 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 1e-6 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --gae_lambda 1.0 \
    --gamma 1.0 \
    --value_loss_coef 0.0 \
    --vf_clip_range 0.2 \
    --use_sae \
    --sae_threshold 0.2 \
    --use_generative_value_model \
    --gen_value_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --gen_value_vllm_num_engines 2 \
    --gen_value_vllm_tensor_parallel_size 1 \
    --gen_value_segmentation sae \
    --gen_value_max_segments 16 \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 1024 \
    --gen_value_conditioning none \
    --gen_value_learning_rate 1e-6 \
    --gen_value_sync_freq 1
