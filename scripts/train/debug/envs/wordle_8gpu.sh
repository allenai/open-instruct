#!/bin/bash
# Beaker experiment for WordleTextEnv — 8 GPUs (2 learners, 6 inference engines)
#
# Text-based Wordle environment: model guesses 5-letter words via <guess> tags.
# Uses PrimeIntellect/Qwen3-1.7B-Wordle-SFT. Config matched to prime-rl example.
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/wordle_8gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "WordleTextEnv 8-GPU (2L/6E) Qwen3-1.7B Wordle" \
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
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-wordle-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 64 \
    --num_samples_per_prompt_rollout 16 \
    --model_name_or_path PrimeIntellect/Qwen3-1.7B-Wordle-SFT \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 204800 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 6 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 42 \
    --inflight_updates True \
    --async_steps 1 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --save_traces \
    --max_steps 20 \
    --tool_parser_type vllm_hermes \
    --dataset_skip_cache \
    --reward_aggregator last \
    --advantage_normalization_type centered \
    --vllm_enable_prefix_caching \
    --output_dir output/wordle_8gpu
