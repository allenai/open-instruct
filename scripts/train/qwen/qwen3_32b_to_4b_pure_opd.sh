#!/bin/bash
# Pure on-policy distillation: Qwen3-32B teacher -> Qwen3-4B student, one 8-GPU node.
# No task rewards (--opd_use_task_rewards false): the student samples on-policy rollouts
# and is trained only on the teacher-top-k forward KL. GPU layout: 4 learner GPUs +
# 2 rollout vLLM engines + Qwen3-32B teacher at TP=2.
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test-codex-opd-olmo-core-grpo}"
shift || true

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Pure OPD: Qwen3-32B teacher -> Qwen3-4B student." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 6h \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env TORCH_COMPILE_DISABLE=1 \
       --gpus 8 \
       --no_auto_dataset_cache \
       --artifact_ttl 7d \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name qwen3_32b_to_4b_pure_opd \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 4096 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 2048 \
    --eval_response_length 2048 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-4B \
    --system_prompt_override_file scripts/train/qwen/math_system_prompt.txt \
    --apply_verifiable_reward false \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 6400 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --beta 0.0 \
    --load_ref_policy false \
    --seed 1 \
    --local_eval_every -1 \
    --checkpoint_state_freq 25 \
    --opd_enabled true \
    --opd_use_task_rewards false \
    --opd_loss_mode forward_kl_topk \
    --opd_topk 16 \
    --opd_loss_coef 1.0 \
    --opd_teacher_model_name_or_path Qwen/Qwen3-32B \
    --opd_teacher_num_engines 1 \
    --opd_teacher_tensor_parallel_size 2 \
    --opd_teacher_gpu_memory_utilization 0.7 \
    --opd_teacher_dtype bfloat16 \
    --with_tracking \
    --push_to_hub false \
    --try_auto_save_to_beaker false "$@"
