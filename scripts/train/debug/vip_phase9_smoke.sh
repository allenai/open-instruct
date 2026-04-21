#!/bin/bash
# Phase 9 smoke test: grpo_fast_genvalue.py end-to-end run on 2 GPUs.
#
# Uses --gen_value_vllm_num_engines 0 so no third GPU is needed; the gen-value
# vLLM pool creation and background scoring thread are skipped, but all other
# machinery (config parsing, resource logging, policy training) is exercised.
#
# For a full 3-GPU test (gen-value pool active), set --gen_value_vllm_num_engines 1
# and ensure a third GPU is available.
set -euo pipefail

unset LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

OUTDIR=/tmp/vip_phase9_output
mkdir -p "$OUTDIR"

.venv/bin/ray stop --force 2>/dev/null || true
.venv/bin/ray start --head --port=8888 --dashboard-host=0.0.0.0
trap ".venv/bin/ray stop --force" EXIT
mkdir -p "$HOME/.triton/autotune"

.venv/bin/python open_instruct/grpo_fast_genvalue.py \
    --exp_name vip_phase9_smoke \
    --output_dir "$OUTDIR" \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 --response_length 512 --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --add_bos --stop_strings "</answer>" \
    --apply_r1_style_format_reward --apply_verifiable_reward true \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --temperature 0.7 --beta 0.0 \
    --learning_rate 3e-7 --total_episodes 160 \
    --deepspeed_stage 2 --num_epochs 1 \
    --num_learners_per_node 1 --vllm_num_engines 1 --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo --vllm_gpu_memory_utilization 0.4 --vllm_enforce_eager \
    --inflight_updates True --async_steps 2 --seed 3 \
    --local_eval_every 5 --save_freq 5 \
    --gradient_checkpointing --with_tracking False --push_to_hub False \
    --use_value_model \
    --value_learning_rate 5e-6 --gae_lambda 0.95 --gamma 1.0 \
    --value_loss_coef 0.5 --vf_clip_range 0.2 \
    --use_generative_value_model \
    --gen_value_vllm_num_engines 0 \
    --gen_value_segmentation fixed \
    --gen_value_chunk_size 256 \
    --gen_value_score_min 0 --gen_value_score_max 10 \
    --gen_value_conditioning none

echo "Phase 9 smoke test: PASS"
