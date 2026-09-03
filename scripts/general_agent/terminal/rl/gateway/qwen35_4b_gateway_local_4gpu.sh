#!/bin/bash

# Local 4-GPU terminal RL through the LiteRegistry gateway backend — no Beaker/mason.py,
# and no podman anywhere on this machine: every sandbox container runs on the remote
# replica fleet behind $GATEWAY_URL.
# Layout: 2 learner GPUs (ZeRO-3 + liger loss) + 2 vLLM engine GPUs. Sized for 4x L40S 46GB.
# Same training args as qwen35_4b_gateway_smoke.sh with a short response length so a
# handful of steps finishes quickly; this validates the plumbing, not model quality.

export GATEWAY_URL="${GATEWAY_URL:-http://jupiter-cs-aus-148.reviz.ai2.in:45216}"
MODEL=Qwen/Qwen3.5-4B
EXP_NAME="${EXP_NAME:-qwen35_4b_gateway_local_4gpu}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0   # ray workers crash re-wrapping themselves in `uv run` here
export SWERL_SANDBOX_TIMING_LOGS=1

mkdir -p "$HOME/.triton/autotune"

ray stop --force
ray start --head --port=8888 --dashboard-host=0.0.0.0

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 4096 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 2 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $MODEL \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 192 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.80 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --tools swerl_vanillux_sandbox \
    --tool_configs "{\"backend\": \"gateway\", \"gateway_url\": \"$GATEWAY_URL\", \"task_data_hf_repo\": \"allenai/tmax-15k-open-instruct\", \"test_timeout\": 120, \"image\": \"python:3.12-slim\"}" \
    --pool_size 64 \
    --max_steps 32 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --inflight_updates true \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --output_dir output/$EXP_NAME \
    --exp_name $EXP_NAME \
    --local_eval_every 10 \
    --save_freq 100
