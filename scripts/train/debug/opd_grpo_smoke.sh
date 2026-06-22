#!/bin/bash
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "OLMo-core GRPO + OPD smoke test." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 30m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env TORCH_COMPILE_DISABLE=1 \
       --gpus 2 \
       --no_auto_dataset_cache \
       --artifact_ttl 1d \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name opd_grpo_smoke \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 256 \
    --response_length 64 \
    --eval_response_length 64 \
    --pack_length 320 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --filter_zero_std_samples false \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --system_prompt_override_file scripts/train/qwen/math_system_prompt.txt \
    --apply_verifiable_reward true \
    --learning_rate 1e-6 \
    --total_episodes 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy false \
    --seed 3 \
    --local_eval_every -1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --opd_enabled true \
    --opd_use_task_rewards true \
    --opd_loss_mode forward_kl_topk \
    --opd_topk 16 \
    --opd_loss_coef 1.0 \
    --opd_teacher_model_name_or_path Qwen/Qwen3-0.6B \
    --opd_teacher_num_engines 1 \
    --opd_teacher_tensor_parallel_size 1 \
    --opd_teacher_gpu_memory_utilization 0.35 \
    --opd_teacher_enforce_eager true \
    --opd_teacher_dtype bfloat16 \
    --push_to_hub false \
    --try_auto_save_to_beaker false "$@"
