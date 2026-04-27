#!/bin/bash

# Sample 100 prompts from hamishivi/swerl-tmax-10k, do 8 rollouts each through the
# full grpo_fast.py vLLM + tool agent loop, and dump every rollout (with
# finish_reason) to /output/rollouts/*.jsonl.
#
# Then run scripts/tmax/debug/count_truncations.py against the dumped JSONL to
# get a truncation breakdown.
#
# We do exactly one training step (--max_steps 1) and zero out the LR so the
# generated rollouts come from the unmodified base policy. Same vllm/tool/sampling
# config as the regular tmax-10k run, so finish_reasons are directly comparable.
# 4 nodes x 8 GPUs (32 GPUs total).

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL tmax-10k truncation counter (1 step, 100 prompts x 8 rollouts)" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 4 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=hamishi740 \
       --secret DOCKER_PAT=hamishivi_DOCKER_PAT \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 100 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --temperature 1.0 \
    --learning_rate 0.0 \
    --total_episodes 800 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 8 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enforce_eager \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 1 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --backend_timeout 1200 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name swerl_qwen35_9b_count_truncations \
    --local_eval_every -1 \
    --save_freq -1 \
    --try_launch_beaker_eval_jobs_on_weka False
