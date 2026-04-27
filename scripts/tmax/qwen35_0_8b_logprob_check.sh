#!/bin/bash
export BEAKER_ALLOW_SUBCONTAINERS=1
export BEAKER_SKIP_DOCKER_SOCKET=1

# Compare training debug/vllm_vs_local_logprob_diff_mean against the offline
# kernel-mismatch measurement from diagnose_logprobs.py / diagnose_multiturn_logprobs.py
#
# Uses Qwen/Qwen3.5-0.8B (hybrid, 24 layers) on a single GPU with async_steps=1
# (minimum lag) so that step-1 vllm_logprobs and local_logprobs use the same weights,
# giving a pure kernel-mismatch reading.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/tmax/qwen35_0_8b_logprob_check.sh

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Qwen3.5-0.8B single-GPU logprob diff check (kernel mismatch vs training metric)" \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 30m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --gpus 1 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 256 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --async_steps 1 \
    --inflight_updates false \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --add_bos \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 512 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --seed 42 \
    --local_eval_every 5 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --exp_name qwen35_0_8b_logprob_check \
    --output_dir /output/qwen35_0_8b_logprob_check
