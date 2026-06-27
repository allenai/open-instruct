#!/bin/bash

# Local 2-GPU debug run for AppWorld RL (GRPO + appworld env).
# Layout: 1 learner GPU + 1 vLLM engine GPU. Qwen3-0.6B + small dataset slice.
# Docker must be running locally; each rollout starts an AppWorld container.
# No Podman services, no Beaker, no mason.py, no HuggingFace.
#
# Prereqs (one-time):
#   1. AppWorld image reachable by the local docker daemon. Either
#        docker pull ghcr.io/stonybrooknlp/appworld:latest   (+ a daemon-visible data root), or
#        build a data-baked image (no bind mount needed):
#          FROM ghcr.io/stonybrooknlp/appworld:latest
#          COPY data /run/data && RUN mkdir -p /run/experiments/outputs
#        docker build -t appworld-data:test <appworld_root>
#   2. A local RL parquet built by scripts/data/convert_appworld_to_rl.py:
#        uv run python scripts/data/convert_appworld_to_rl.py \
#          --data_root <appworld_root> --split train --limit 16 --max_steps 10 \
#          --output_parquet $APPWORLD_DEBUG_PARQUET

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

# Data-baked image (data_root="" => no bind mount) and local debug dataset.
APPWORLD_IMAGE="${APPWORLD_IMAGE:-appworld-data:test}"
APPWORLD_DEBUG_PARQUET="${APPWORLD_DEBUG_PARQUET:-/tmp/aw_probe/appworld_debug16.parquet}"

uv run python open_instruct/grpo_fast.py \
    --exp_name appworld_local_rl_2gpu \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mixer_list "${APPWORLD_DEBUG_PARQUET}" 16 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 16 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools appworld \
    --tool_call_names execute_python \
    --tool_configs "{\"image\": \"${APPWORLD_IMAGE}\", \"data_root\": \"\", \"max_interactions\": 10}" \
    --tool_parser_type vllm_qwen3_xml \
    --pool_size 4 \
    --max_steps 10 \
    --backend_timeout 600 \
    --gradient_checkpointing \
    --save_traces \
    --local_eval_every 8 \
    --logging_steps 1 \
    --seed 42 \
    --report_to null \
    --output_dir output/appworld_rl_local_2gpu \
    --push_to_hub false
