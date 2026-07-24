#!/bin/bash

# Quick 2-GPU smoke test for reinforce_ada_est on the OLMo-core GRPO path (open_instruct/grpo.py).
# Verifies that per-prompt completions-per-rollout derived from the `pass_count` column (dataset
# quartiles) runs end to end without crashing. Uses a small model and a handful of steps.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${BEAKER_IMAGE:-${BEAKER_USER}/open-instruct-integration-test-ngu}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Quick 2-GPU reinforce_ada_est smoke test on deepscaler quartiles (grpo.py)." \
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
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name reinforce_ada_est_2gpu \
    --dataset_mixer_list mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 1024 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 16 \
    --reinforce_ada_est True \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --inflight_updates True \
    --active_sampling \
    --advantage_normalization_type centered \
    --learning_rate 1e-6 \
    --total_episodes 256 \
    --fsdp_shard_degree 1 \
    --fsdp_num_replicas 1 \
    --activation_memory_budget 0.5 \
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
    --push_to_hub false "$@"
