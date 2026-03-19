#!/bin/bash
# Short hybrid GRPO run to capture packed vs unpacked logprob diff metrics.
# Runs ~5 steps on a single node (8 GPUs) with the production model/config,
# but fewer prompts and episodes to finish quickly.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run --no-sync python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --description "Hybrid packed metrics debug run" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 60m \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name hybrid_packed_metrics_debug \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 2 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list hamishivi/rlvr_acecoder_filtered_filtered 100 hamishivi/omega-combined-no-boxed_filtered 100 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/omega-combined 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 11264 \
    --model_name_or_path allenai/Olmo-Hybrid-Instruct-DPO-7B \
    --trust_remote_code \
    --vllm_enforce_eager \
    --chat_template_name olmo123 \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 500 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --inflight_updates true \
    --async_steps 4 \
    --active_sampling \
    --advantage_normalization_type centered \
    --save_traces \
    --wandb_entity ai2-llm
