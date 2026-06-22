#!/bin/bash
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true

EXP_NAME="${EXP_NAME:-opd_grpo_qwen3_0p6b_teacher4b_topk128_500steps}"
PRIORITY="${PRIORITY:-urgent}"

echo "Using Beaker image: ${BEAKER_IMAGE}"
echo "Experiment name: ${EXP_NAME}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --cluster ai2/saturn \
       --image "${BEAKER_IMAGE}" \
       --description "OLMo-core GRPO + OPD scale rehearsal: Qwen3 0.6B student, Qwen3 4B teacher, top-k 128, 500 steps." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority "${PRIORITY}" \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 4h \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env TORCH_COMPILE_DISABLE=1 \
       --gpus 2 \
       --no_auto_dataset_cache \
       --artifact_ttl 7d \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name "${EXP_NAME}" \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 4096 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 128 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --eval_response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --filter_zero_std_samples false \
    --async_steps 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --system_prompt_override_file scripts/train/qwen/math_system_prompt.txt \
    --apply_verifiable_reward true \
    --with_tracking \
    --wandb_project open_instruct_internal \
    --wandb_entity ai2-llm \
    --learning_rate 1e-6 \
    --total_episodes 16000 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy false \
    --seed 3 \
    --local_eval_every 100 \
    --save_freq 500 \
    --checkpoint_state_freq 500 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --opd_enabled true \
    --opd_use_task_rewards true \
    --opd_loss_mode forward_kl_topk \
    --opd_topk 128 \
    --opd_loss_coef 1.0 \
    --opd_teacher_model_name_or_path Qwen/Qwen3-4B \
    --opd_teacher_num_engines 1 \
    --opd_teacher_tensor_parallel_size 1 \
    --opd_teacher_gpu_memory_utilization 0.75 \
    --opd_teacher_enforce_eager true \
    --opd_teacher_dtype bfloat16 \
    --push_to_hub false \
    --try_auto_save_to_beaker false "$@"
