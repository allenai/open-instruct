#!/bin/bash
# Debug script for Harbor-based GRPO training on OpenThoughts-TBLite.
# Runs 1 training node (8 GPUs) + 1 inference node (8 GPUs)
# using Harbor's Docker backend with Terminus-2 for agent rollouts.
#
# Dataset: open-thoughts/OpenThoughts-TBLite (100 terminal-agent tasks).
# The harbor_tokenize_v1 transform maps task_name → harbor_task_path
# and instruction → prompt.
#
# Prerequisites:
#   - The Docker image must have `harbor` installed (`pip install harbor`).
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/harbor_grpo.sh

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "Harbor GRPO debug — OpenThoughts-TBLite" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list NousResearch/openthoughts-tblite 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list NousResearch/openthoughts-tblite 8 \
    --dataset_mixer_eval_list_splits train \
    --dataset_transform_fn harbor_tokenize_v1 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 32768 \
    --inflight_updates True \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 0.7 \
    --exp_name harbor_grpo_tblite \
    --learning_rate 3e-7 \
    --total_episodes $((5 * 8 * 4)) \
    --deepspeed_stage 3 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 1 \
    --local_eval_every 10 \
    --gradient_checkpointing \
    --push_to_hub false \
    --output_dir /output \
    --kl_estimator 2 \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --vllm_enable_prefix_caching \
    --use_harbor \
    --harbor_agent_name terminus-2 \
    --harbor_environment docker
