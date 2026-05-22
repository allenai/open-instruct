#!/bin/bash
# Multi-task RL — Approach (3) cascade, STAGE 2: swerl_sandbox RL, anchored to stage 1.
#
# Resumes from a stage-1 checkpoint produced by cascade_stage1_drtulu.sh. The stage-1
# checkpoint serves a dual role:
#   1. Starting weights for stage 2 (--model_name_or_path STAGE1_CHECKPOINT).
#   2. Frozen reference policy for KL anti-forgetting (--load_ref_policy True --beta 0.005).
#
# By default --ref_policy_update_freq is null, so the reference stays frozen for the
# whole stage. That's what gives the anti-forgetting pressure — every gradient step
# pulls the policy back toward stage-1 behavior on stage-2 prompts.
#
# Tune --beta: higher → stronger preservation of stage-1 behavior, slower stage-2 learning.
# Typical range 0.001-0.02 for KL-anchor cascades.
#
# Required env vars:
#   STAGE1_CHECKPOINT — path/HF-name of the final stage-1 checkpoint.
#
# Launch:
#   STAGE1_CHECKPOINT=/path/to/cascade_stage1/output ./scripts/train/build_image_and_launch.sh \
#     scripts/multi_task_rl/cascade_stage2_swerl_with_kl_anchor.sh

EXP_NAME="${EXP_NAME:-cascade_stage2_swerl_anchored}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${STAGE1_CHECKPOINT:-}" ]]; then
    echo "ERROR: set STAGE1_CHECKPOINT to the final checkpoint from cascade_stage1_drtulu.sh" >&2
    exit 1
fi

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "${RUN_NAME}" \
       --pure_docker_mode \
       --workspace ai2/general-tool-use \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=8 \
       --env SWERL_DOCKER_START_CONCURRENCY=128 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env MIRROR_URL=jupiter-cs-aus-150.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --budget ai2/oe-omai \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    \
    `# >>> CASCADE STAGE 2 CORE: start from stage 1, anchor KL to it. <<<` \
    --model_name_or_path "${STAGE1_CHECKPOINT}" \
    --load_ref_policy True \
    --beta 0.005 \
    `# --ref_policy_update_freq null  # default = no EMA update, ref stays frozen.` \
    `# To soften the anchor over time, uncomment and set e.g. --ref_policy_update_freq 100.` \
    \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 35840 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 5120 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 128 \
    --max_steps 50 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --local_eval_every 10 \
    --save_freq 50 \
    --keep_last_n_checkpoints -1 \
    --try_launch_beaker_eval_jobs_on_weka False
