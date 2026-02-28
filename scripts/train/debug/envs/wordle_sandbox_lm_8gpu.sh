#!/bin/bash
# Beaker experiment for WordleTextEnv + SandboxLM tools — 8 GPUs (4 learners, 4 inference engines)
#
# Wordle text environment with sandbox tools available every rollout.
# Uses Wordle dataset and a combined Wordle + SandboxLM system prompt.
# Tuned for stability to avoid filter collapse / actor pressure in debug runs.
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/wordle_sandbox_lm_8gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "WordleTextEnv + SandboxLM 8-GPU debug (4L/4E)" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --preemptible \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-wordle-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path PrimeIntellect/Qwen3-1.7B-Wordle-SFT \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 204800 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 42 \
    --inflight_updates True \
    --async_steps 1 \
    --truncated_importance_sampling_ratio_cap 5.0 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --save_traces \
    --tools wordle generic_sandbox \
    --pool_size 16 \
    --max_steps 20 \
    --per_turn_max_tokens 512 \
    --tool_parser_type vllm_qwen3xml \
    --dataset_skip_cache \
    --no_filter_zero_std_samples \
    --reward_aggregator last \
    --advantage_normalization_type centered \
    --system_prompt_override_file scripts/train/debug/envs/wordle_sandbox_lm_system_prompt.txt \
    --vllm_enable_prefix_caching \
    --output_dir output/wordle_sandbox_lm_8gpu
