#!/bin/bash
# LOCAL (no Beaker) single-GPU GRPO smoke test for the OLMo-hybrid 275M -hf checkpoint.
# Runs grpo_fast.py directly against the local .venv on the current GPU. For quick
# validation that the forks load, vLLM starts, and the first weight sync succeeds.
set -uo pipefail

export REPO_PATH=/weka/oe-adapt-default/michaeln/nit-open-instruct
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH"
export NCCL_CUMEM_ENABLE=0
# The olmo_hybrid_small vLLM output carries a torch.dtype that vLLM V1's msgpack
# encoder can't serialize over the EngineCore socket; allow the pickle fallback.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HOME/.triton/autotune"

ray stop --force >/dev/null 2>&1 || true

MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf

.venv/bin/python open_instruct/grpo_fast.py \
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
    --model_name_or_path "$MODEL_PATH" \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --inflight_updates True \
    --ground_truths_key ground_truth \
    --chat_template_name olmo_thinker \
    --learning_rate 3e-7 \
    --total_episodes 64 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy true \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode
