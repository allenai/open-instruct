#!/bin/bash

# 4-node / 32 GPU DPPO run @ 64k max length (full DPPO_repro recipe) with
# sandboxes on the self-hosted OpenSandbox service on GKE (OpenSandboxBackend).
#
# Experiment config inherited from qwen35_9b_dppo_repro_4node_64k.sh (omni_agent);
# only the sandbox backend plumbing differs: no nested containers, no podman
# services, no on-node registry mirror — sandbox image pulls route through the
# Artifact Registry pull-through cache instead (SWERL_OPENSANDBOX_IMAGE_PREFIX;
# set it empty to pull straight from Docker Hub). Requires the
# pradeepd_OPEN_SANDBOX_API_KEY Beaker secret and outbound egress to the endpoint
# (verify with scripts/opensandbox/check_opensandbox_egress.sh). See
# docs/sandbox_management.md for tuning, Spot behavior, and the killed-job
# zombie-sandbox trap (sweep with the janitor before relaunching after a kill).
#
# OpenSandbox-specific settings, mirroring qwen35_4b_base_tmax_10k_opensandbox.sh:
# - SWERL_OPENSANDBOX_CPU=2: at cpu=1 node contention stretched rollouts ~2.4x.
# - BEAKER_MIN_RUNTIME (default 8h): Beaker won't preempt before the first
#   checkpoint_state saves exist (needs beaker-py >= 2.7.2 for mason --min_runtime).
# - MASK_INFRA_FAILED (default true): exclude Spot-preempted/reset-failed rollouts
#   from GRPO group advantages and the batch; val/infra_failed_rate reports the
#   affected fraction either way.
# - checkpoint_state_dir derives from EXP_NAME: renaming the experiment gets a
#   fresh checkpoint identity, while retries of the same name resume correctly.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

# Loss exclusion on/off is baked into the experiment name and description so the
# two variants are distinguishable in wandb and Beaker (and get separate
# checkpoint dirs via EXP_NAME). Labels are kept short: exp_name is used as a
# wandb tag, which has a hard 64-char limit, and the base name is already long.
MASK_INFRA_FAILED="${MASK_INFRA_FAILED:-true}"
if [ "$MASK_INFRA_FAILED" = "true" ]; then
    MASK_LABEL="mask_infra"
else
    MASK_LABEL="no_mask_infra"
fi
EXP_NAME="${EXP_NAME:-swerl_qwen35_9b_dppo_repro_4node_64k_opensandbox_${MASK_LABEL}}"
# For the B300 cluster:
#   BEAKER_CLUSTER=ai2/holmes BEAKER_WORKSPACE=ai2/oe-agents-holmes $0 <image>
# with an image built via `build_image_and_launch.sh --cuda-version 13`; the
# Beaker secrets (pradeepd_OPEN_SANDBOX_API_KEY, pradeepd_DOCKER_PAT,
# pradeepd_WANDB_API_KEY) must exist in that workspace too, and OpenSandbox
# egress should be verified once from a holmes session
# (scripts/opensandbox/check_opensandbox_egress.sh).
BEAKER_CLUSTER="${BEAKER_CLUSTER:-ai2/holmes}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents-holmes}"

uv run python mason.py \
       --cluster "$BEAKER_CLUSTER" \
       --image "$BEAKER_IMAGE" \
       --description "tmax-15k DPPO Qwen35 9b (repro; 4-node; 64k; OpenSandbox spot sandboxes; ${MASK_LABEL})" \
       --pure_docker_mode \
       --workspace "$BEAKER_WORKSPACE" \
       --priority "${BEAKER_PRIORITY:-high}" \
       --preemptible \
       --min_runtime "${BEAKER_MIN_RUNTIME:-8h}" \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_OPENSANDBOX_DOMAIN="${SWERL_OPENSANDBOX_DOMAIN:-sandbox.oe-rl-sandbox.apps.allenai.org}" \
       --env SWERL_OPENSANDBOX_PROTOCOL="${SWERL_OPENSANDBOX_PROTOCOL:-https}" \
       --env SWERL_OPENSANDBOX_LIFETIME_S=3600 \
       --env SWERL_OPENSANDBOX_START_CONCURRENCY="${SWERL_OPENSANDBOX_START_CONCURRENCY:-64}" \
       --env SWERL_OPENSANDBOX_CPU="${SWERL_OPENSANDBOX_CPU:-2}" \
       --env SWERL_OPENSANDBOX_IMAGE_PREFIX="${SWERL_OPENSANDBOX_IMAGE_PREFIX:-us-docker.pkg.dev/ai2-skiff2-oe-rl-sandbox/docker-hub-remote-repository}" \
       --env SWERL_OPENSANDBOX_APP_NAME=swerl-tmax-9b-opensandbox \
       --env DOCKERHUB_USERNAME=pdasigi \
       --secret DOCKER_PAT=pradeepd_DOCKER_PAT \
       --secret OPEN_SANDBOX_API_KEY=pradeepd_OPEN_SANDBOX_API_KEY \
       --secret WANDB_API_KEY=pradeepd_WANDB_API_KEY \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 16384 \
    --response_length 65536 \
    --pack_length 67584 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 32 \
    --async_steps 4 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
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
    --seed 42 \
    --gradient_checkpointing \
    --attn_implementation torch \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"backend": "opensandbox", "task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 1024 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --mask_infra_failed_completions "$MASK_INFRA_FAILED" \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_freq 10 \
    --checkpoint_state_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/${EXP_NAME}" \
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
    --exp_name "$EXP_NAME" \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False \
    \; bash scripts/opensandbox/cleanup_opensandbox_sandboxes.sh swerl-tmax-9b-opensandbox
