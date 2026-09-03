#!/bin/bash

# 4-node / 32 GPU DPPO prod run @ 64k on holmes (cu13/B300), sandboxes on the LiteRegistry
# podman fleet behind $GATEWAY_URL instead of podman services colocated in the job.
# Training recipe is identical to qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh; the only
# change is `"backend": "gateway"` in --tool_configs, which makes every SWERL_PODMAN_* /
# MIRROR_URL / DOCKER_PAT / docker_login.sh / subcontainer setting unnecessary.
# Preflight the deployment first:
#   uv run python scripts/general_agent/terminal/rl/gateway/check_gateway_deployment.py --gateway_url $GATEWAY_URL

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

GATEWAY_URL="${GATEWAY_URL:-http://jupiter-cs-aus-148.reviz.ai2.in:45216}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

EXP_NAME=swerl_qwen35_9b_dppo_prod_4node_64k_holmes_gateway

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "tmax-15k DPPO Qwen35 9b (repro; 4-node; 64k; holmes/cu13/B300; sandboxes via literegistry gateway jupiter-cs-aus-148:45216, 32 replicas)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents-holmes \
       --priority urgent \
       --preemptible \
       --min_runtime 28800s \
       --auto_resume \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
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
    --attn_implementation flash_4 \
    --num_epochs 1 \
    --num_learners_per_node 8 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs "{\"backend\": \"gateway\", \"gateway_url\": \"$GATEWAY_URL\", \"task_data_hf_repo\": \"allenai/tmax-15k-open-instruct\", \"test_timeout\": 120, \"image\": \"python:3.12-slim\"}" \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/qwen35_9b_prod_4node_64k_holmes_gateway_001 \
    --checkpoint_state_freq 5 \
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
