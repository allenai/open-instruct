#!/bin/bash

# Smoke test for the LiteRegistry Podman *gateway* sandbox backend.
#
# Runs a small terminal RL job (Qwen3.5-4B, 4 GPUs: 2 learners + 2 vLLM engines)
# whose sandboxes are remote containers behind a LiteRegistry gateway
# (backend=gateway) instead of podman services colocated in this job.
# Note what is absent compared to the podman-colocated scripts: no
# docker_login.sh, no SWERL_PODMAN_* / MIRROR_URL / DOCKER_PAT /
# BEAKER_ALLOW_SUBCONTAINERS plumbing.
#
# Usage: ./scripts/train/build_image_and_launch.sh scripts/general_agent/terminal/rl/gateway/qwen35_4b_gateway_smoke_2gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

# LiteRegistry gateway (redis + gateway + podman replicas + docker mirrors).
GATEWAY_URL=http://neptune-cs-aus-265.reviz.ai2.in:56992

MODEL=Qwen/Qwen3.5-4B
EXP_NAME=swerl_qwen35_4b_gateway_smoke

uv run python mason.py \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Gateway-backend terminal RL smoke (Qwen3.5-4B; 2 GPU; LiteRegistry podman gateway)" \
       --pure_docker_mode \
       --workspace ai2/general-tool-use \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --gpus 4 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 8192 \
    --response_length 16384 \
    --pack_length 18432 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 2 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $MODEL \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 320 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
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
    --tools swerl_vanillux_sandbox \
    --tool_configs "{\"backend\": \"gateway\", \"gateway_url\": \"$GATEWAY_URL\", \"task_data_hf_repo\": \"allenai/tmax-15k-open-instruct\", \"test_timeout\": 120, \"image\": \"python:3.12-slim\"}" \
    --pool_size 64 \
    --max_steps 32 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --inflight_updates true \
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
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False
