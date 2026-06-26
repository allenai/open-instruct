#!/bin/bash

# Quick 2-GPU never-give-up (NGU) smoke test on the DeepScaleR difficulty-quartile dataset.
# Verifies that math_* verifier sources resolve to the math verifier and that the per-dataset
# (per-quartile) batch metrics are logged. Uses a small model and a handful of steps.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test-ngu}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Quick 2-GPU NGU smoke test on deepscaler quartiles." \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 30m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --gpus 2 \
       --no_auto_dataset_cache \
       --artifact_ttl 1d \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name ngu_quartiles_2gpu \
    --dataset_mixer_list mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 1024 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --inflight_updates True \
    --active_sampling \
    --never_give_up 1.0 \
    --advantage_normalization_type centered \
    --learning_rate 1e-6 \
    --total_episodes 256 \
    --deepspeed_stage 2 \
    --beta 0.0 \
    --load_ref_policy False \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --seed 1 \
    --local_eval_every 1000000 \
    --with_tracking \
    --push_to_hub false
