#!/bin/bash

# 64k-context variant of recreate_blowup_mask.sh that also starts from the vanilla
# Qwen/Qwen3.5-9B base (rather than the pre-blowup checkpoint) so we can see whether
# CISPO + the trust-region mask produces a stable long-context run from scratch.
# Sequence parallel is 4 (vs. 2) to keep per-GPU activation memory flat.
# 4 nodes x 8 GPUs (32 GPUs total); DP per node = 8 / sp=4 = 2.
#
# Response + prompt fit:
#   max_prompt_token_length 2048 + response_length 65536 = 67584 min.
#   pack_length 68608 keeps the same ~1k slack that 35840 had over 34816.
#
# Mask mapping (unchanged from recreate_blowup_mask.sh):
#   --tis_mask_lower 0.5 -> ratio lower bound = 0.5
#   --tis_mask_upper 3.0 -> ratio upper bound = 3.0
#   --clip_higher    2.0 -> keeps CISPO's internal clamp at the mask upper.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL tmax-10k GRPO with Qwen3.5-9B + CISPO mask, 64k ctx, sp=4" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=hamishi740 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=8 \
       --env SWERL_DOCKER_START_CONCURRENCY=128 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env MIRROR_URL=jupiter-cs-aus-150.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=hamishivi_DOCKER_PAT \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/swerl-tmax-15k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 8192 \
    --response_length 65536 \
    --pack_length 68608 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 128000 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 8 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --loss_fn cispo \
    --clip_higher 2.0 \
    --tis_mask_lower 0.5 \
    --tis_mask_upper 3.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enforce_eager \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-15k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --mask_truncated_completions true \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name swerl_qwen35_9b_base_tmax_10k_grpo_mask_64k \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
