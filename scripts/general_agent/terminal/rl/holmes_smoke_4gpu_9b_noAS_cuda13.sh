#!/bin/bash

# 4-GPU Terminal RL SMOKE TEST on ai2/holmes (CUDA 13 / B300) with a REAL 9B model.
# Same production recipe as holmes_smoke_4gpu_cuda13.sh (swerl_vanillux_sandbox +
# allenai/tmax-15k-open-instruct + DPPO + SP=2 + flash_2), but the policy is the
# tmax Qwen3.5-9B SFT+DPPO checkpoint (step_360, _cg-converted) instead of Qwen3-0.6B.
# This variant has active_sampling OFF: it accumulates exactly the batch and steps
# (no oversampling for reward variance), so a training step completes fast/deterministically
# regardless of model quality. Companion to holmes_smoke_4gpu_9b_cuda13.sh (active_sampling ON).
#
# Model = the DPPO-9B step_360 checkpoint, CG-converted for vLLM serving. Its weka path
# contains "qwen35", so the trainer->vLLM weight-sync name-mapper (_build_vlm_name_mapper)
# triggers correctly. (allenai/tmax-9b would NOT — its name matches no qwen spelling — so
# it'd need an architecture-aware mapper fix first.) Local path exists → grpo_fast skips
# snapshot_download. Config-only vs the 0.6B smoke → reuse the same cu13 image (no rebuild).
#
# Layout: 4 GPUs / 1 node = 2 learners (SP=2, stage 3) + 2 vLLM engines. Fits on B300 (288GB).
# Launch (reuse image):
#   source env.cuda13.sh
#   bash scripts/general_agent/terminal/rl/holmes_smoke_4gpu_9b_cuda13.sh \
#       shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
MODEL=/weka/oe-adapt-default/allennlp/deletable_checkpoint/shashankg/swerl_qwen35_9b_dppo_repro_4node_64k__42__1784235838_checkpoints/step_360_cg
TOKENIZER=$MODEL
DATASET=allenai/tmax-15k-open-instruct
EXP_NAME=swerl_holmes_smoke_9b_tmax_noAS_cuda13

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "CUDA-13/B300 4-GPU terminal RL smoke, 9B tmax step_360_cg (vanillux, DPPO, SP=2, fa2, NO active_sampling)" \
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
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
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
       --env MIRROR_URL=jupiter-cs-aus-102.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 4 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list $DATASET 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 2048 \
    --response_length 4096 \
    --pack_length 8192 \
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
