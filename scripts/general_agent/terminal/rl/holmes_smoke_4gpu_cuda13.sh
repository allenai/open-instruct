#!/bin/bash

# 4-GPU Terminal RL SMOKE TEST on ai2/holmes (CUDA 13 / B300).
# Purpose: validate the cu13 image + holmes cluster + the PRODUCTION terminal-RL
# recipe (swerl_vanillux_sandbox + allenai/tmax-15k-open-instruct + DPPO loss) end
# to end on Beaker, AND exercise sequence parallelism.
#
# Mirrors the production script scripts/general_agent/terminal/rl/qwen35_9b_dppo_repro.sh
# (tool/dataset/DPPO/liger/lm_head_fp32/envs) but shrunk to a smoke:
#   - small model (Qwen3-0.6B) + small context to avoid OOM
#   - 4 GPUs on 1 node = 2 learner GPUs (SP=2, stage 3) + 2 vLLM engine GPUs
#   - ~2 training steps
# NOT a real training run.
#
# Launch via:  ./scripts/train/build_image_and_launch_dirty.sh --cuda-version 13 \
#                  scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh
#
# Submit-side note: `uv run` pins the cuda13 group so mason.py runs on a CUDA-13 box.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
MODEL=Qwen/Qwen3-0.6B
TOKENIZER=Qwen/Qwen3-0.6B
DATASET=allenai/tmax-15k-open-instruct
EXP_NAME=swerl_holmes_smoke_4gpu_cuda13_dppo

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "CUDA-13/B300 4-GPU terminal RL smoke (Qwen3-0.6B, vanillux, DPPO, SP=2)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 1 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=4 \
       --env SWERL_DOCKER_START_CONCURRENCY=64 \
       --env SWERL_DOCKER_EXEC_CONCURRENCY=256 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_PODMAN_IMAGE_JANITOR_ENABLED=1 \
       --env SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S=60 \
       --env SWERL_PODMAN_IMAGE_JANITOR_UNTIL=10m \
       --env MIRROR_URL=jupiter-cs-aus-112.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 4 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list $DATASET 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --per_turn_max_tokens 2048 \
    --response_length 4096 \
    --pack_length 6144 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --async_steps 2 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 32 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --attn_implementation flash_2 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_enable_prefix_caching \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 64 \
    --max_steps 4 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --lm_head_fp32 true \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --output_dir /output \
    --exp_name $EXP_NAME \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
