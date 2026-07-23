#!/bin/bash

# RL on Qwen/Qwen3.5-4B + hamishivi/swerl-tmax-15k
# 2 nodes x 8 GPUs (16 GPUs total)
#
# Same as qwen35_4b_base_tmax_10k_8_podman_services_2node_toy.sh, but runs
# sandboxes on a self-hosted OpenSandbox service on GKE Autopilot
# (OpenSandboxBackend) instead of on-node Podman — no nested containers, no
# podman services, no registry mirror. Requires the
# pradeepd_OPEN_SANDBOX_API_KEY Beaker secret, SWERL_OPENSANDBOX_DOMAIN set in
# the launching shell, and outbound egress to that endpoint (verify with
# scripts/opensandbox/check_opensandbox_egress.sh). See
# docs/sandbox_management.md for the trade-offs vs Podman and Modal.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
: "${SWERL_OPENSANDBOX_DOMAIN:?Set SWERL_OPENSANDBOX_DOMAIN to the OpenSandbox endpoint (e.g. sandbox.example.com)}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL tmax-10k GRPO with Qwen3.5-4B pool size 128 (OpenSandbox sandboxes)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_OPENSANDBOX_DOMAIN="$SWERL_OPENSANDBOX_DOMAIN" \
       --env SWERL_OPENSANDBOX_PROTOCOL="${SWERL_OPENSANDBOX_PROTOCOL:-http}" \
       --env SWERL_OPENSANDBOX_LIFETIME_S=3600 \
       --env SWERL_OPENSANDBOX_APP_NAME=swerl-tmax-opensandbox-toy \
       --secret OPEN_SANDBOX_API_KEY=pradeepd_OPEN_SANDBOX_API_KEY \
       --secret WANDB_API_KEY=pradeepd_WANDB_API_KEY \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/swerl-tmax-15k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 16384 \
    --response_length 65536 \
    --pack_length 67584 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 1280 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --save_trainer_logprobs false \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"backend": "opensandbox", "task_data_hf_repo": "hamishivi/swerl-tmax-15k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 128 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --output_dir /output \
    --exp_name swerl_qwen35_4b_base_tmax_grpo_15k_opensandbox \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False \
    \; bash scripts/opensandbox/cleanup_opensandbox_sandboxes.sh swerl-tmax-opensandbox-toy
