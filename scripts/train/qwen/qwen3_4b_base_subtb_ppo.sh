#!/bin/bash

# Qwen3-4B-Base proper SubTB+GM training run.
# Uses the dedicated SubTB algorithm mode rather than PPO-style policy updates.

DDMM=$(date +"%d%m")
EXP_NAME="${EXP_NAME:-vip_subtb_${DDMM}_qwen3_4b_math}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Base"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
if [ $# -gt 0 ]; then
    shift
fi

uv run mason.py \
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
    --env WANDB_RUN_ID=${RUN_NAME} \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --algorithm_mode subtb \
    --beta 0.0 \
    --async_steps 1 \
    --inflight_updates \
    --no_resampling_pass_rate 0.875 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 0.6666667 \
    --total_episodes 128000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_top_p 1.0 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/${RUN_NAME} \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --eval_on_step_0 True \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --oe_eval_max_length 10240 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero \
    --oe_eval_gpu_multiplier 4 \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    --use_value_model \
    --value_loss_type subtb_gm \
    --value_learning_rate 1e-6 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --value_loss_coef 0.5 \
    --subtb_q 0.5 \
    --subtb_alpha 2.0 \
    --subtb_omega 1.0 \
    --subtb_reward_scale 10.0 \
    --subtb_num_windows 4 \
    --subtb_min_window_size 16 \
    --subtb_max_window_size 256 \
    --subtb_lambda 0.9 "$@"
