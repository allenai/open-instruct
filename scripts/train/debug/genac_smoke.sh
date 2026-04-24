#!/bin/bash
# Minimal smoke test for the generative-value (GenAC) path in grpo_fast_genvalue.py.
#
# Exercises end-to-end:
#   1. Policy learner + policy vLLM engine (colocated via --single_gpu_mode, IPC).
#   2. Second vLLM pool hosting the generative critic.
#   3. GenValueTrainerActor (PyTorch copy of the critic) running REINFORCE.
#   4. NCCL weight sync from trainer actor → critic vLLM pool every policy step
#      (gen_value_sync_freq=1).
#
# Compute footprint: 3 GPUs minimum.
#   GPU 0: policy learner (DeepSpeed) + policy vLLM engine (shared via IPC)
#   GPU 1: generative-critic vLLM engine
#   GPU 2: generative-critic PyTorch trainer actor (GenValueTrainerActor)
#
# Runs ~3-5 policy steps on Qwen3-0.6B with short prompts / responses so the
# whole loop (rollout → gen-value scoring → GAE with piecewise-constant values →
# REINFORCE step on the critic → NCCL weight broadcast) finishes in a few minutes
# on a 3×A100 / 3×L40S box.
set -euo pipefail

unset LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/tmp/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_home/datasets}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

.venv/bin/ray stop --force 2>/dev/null || true
.venv/bin/ray start --head --port=8888 --dashboard-host=0.0.0.0
trap ".venv/bin/ray stop --force" EXIT

mkdir -p "$HOME/.triton/autotune"

.venv/bin/python open_instruct/grpo_fast_genvalue.py \
    --exp_name genac_smoke \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 256 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 2 \
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
    --total_episodes 32 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --inflight_updates True \
    --async_steps 2 \
    --seed 3 \
    --local_eval_every 3 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking False \
    --push_to_hub False \
    --output_dir /tmp/genac_smoke_output \
    --use_value_model \
    --value_learning_rate 5e-6 \
    --gae_lambda 0.95 \
    --gamma 1.0 \
    --value_loss_coef 0.0 \
    --vf_clip_range 0.2 \
    --use_generative_value_model \
    --gen_value_model_name_or_path Qwen/Qwen3-0.6B \
    --gen_value_vllm_num_engines 1 \
    --gen_value_vllm_tensor_parallel_size 1 \
    --gen_value_segmentation fixed \
    --gen_value_chunk_size 64 \
    --gen_value_max_segments 4 \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 128 \
    --gen_value_conditioning none \
    --gen_value_learning_rate 1e-6 \
    --gen_value_sync_freq 1 \
    "${@}"
