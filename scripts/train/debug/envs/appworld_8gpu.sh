#!/bin/bash
# Training script for GRPO with AppWorld environment
#
# AppWorld: Interactive environment with 9 apps (Spotify, Amazon, Venmo, etc.)
# and 457 APIs. The model executes Python code to complete tasks.
#
# Requirements:
# - appworld package installed (included in project dependencies)
# - 8 GPUs available
#
# Note: This script uses mason to launch on Beaker cluster.
# AppWorld data is downloaded at runtime via `appworld download data`.

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "AppWorld GRPO training with Qwen3-4B" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env APPWORLD_ROOT=. \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run appworld download data \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-appworld-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 4096 \
    --response_length 16384 \
    --pack_length 65536 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 640 \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --env_pool_size 8 \
    --env_max_steps 50 \
    --tool_parser_type vllm_hermes \
    --no_filter_zero_std_samples \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name appworld_qwen3_4b_grpo \
    --local_eval_every 10 \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False
