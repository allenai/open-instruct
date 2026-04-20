#!/bin/bash
# Local 2-GPU smoke test for VIP value model (Phase 1).
# Expects 2 GPUs: 1 learner (DeepSpeed) + 1 vLLM engine.
# Runs ~5 training steps to verify end-to-end: init, value forward, GAE, value backward, checkpoint.
set -euo pipefail

unset LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

.venv/bin/ray stop --force 2>/dev/null || true
.venv/bin/ray start --head --port=8888 --dashboard-host=0.0.0.0
trap ".venv/bin/ray stop --force" EXIT

mkdir -p "$HOME/.triton/autotune"

.venv/bin/python open_instruct/grpo_fast.py \
    --exp_name vip_local_2gpu_smoke \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --add_bos \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --temperature 0.7 \
    --beta 0.0 \
    --learning_rate 3e-7 \
    --total_episodes 160 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_enforce_eager \
    --inflight_updates True \
    --async_steps 2 \
    --seed 3 \
    --local_eval_every 5 \
    --save_freq 5 \
    --gradient_checkpointing \
    --with_tracking False \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 5e-6 \
    --gae_lambda 0.95 \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --output_dir /tmp/vip_smoke_output
