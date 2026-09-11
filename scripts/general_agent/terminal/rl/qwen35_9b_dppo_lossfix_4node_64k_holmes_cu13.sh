#!/bin/bash

# 4-node / 32 GPU DPPO @ 64k on holmes (B300, CUDA 13) -- A/B for the GRPO loss-scale fix.
# Identical recipe to qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh (the prod run
# swerl_qwen35_9b_dppo_prod_4node_64k_holmes); the ONLY intended difference is the code:
# this branch (loss_scale_fix_cuda13) ports hamishivi/tmax 50f22169 "Fix GRPO policy
# gradient scaling with sequence parallelism":
#   - tiled/liger loss: per-rank normalizer * divisor / global_denominator (was: all-reduced
#     mean count -> rank-local normalization that upweighted low-token SP shards)
#   - DeepSpeed ZeRO-3 reduce_scatter divides by the FULL world size (not world/sp), so the
#     "undo averaging" multiplier is world_size under stage 3 (grad_norm will read ~4x higher
#     than the prod run at SP=4 -- expected, not a regression).
# Compare against swerl_qwen35_9b_dppo_prod_4node_64k_holmes in wandb (oe-general-agents).
# RAY_enable_open_telemetry=0: the first launch (01M26TTNZTM1P499HW0B140EJV) wedged on retry with the
# Ray OTel getenv SIGSEGV -> EnvironmentPool.__init__ ActorDiedError -> silent hang; Ray metrics are
# unused (monitoring is wandb), so disable the OTel recorder.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

EXP_NAME=swerl_qwen35_9b_dppo_lossfix_4node_64k_holmes

# Registry mirrors: warmed ones first (144, 143), then the other live ai2/oe-agents mirrors
# (verified 2026-09-10: /v2/ 200 + serve hamishi740/swerl-tmax-v3 tags). 145 was refusing
# connections on 2026-09-10; kept last as a harmless fallback in case it comes back.
MIRRORS=jupiter-cs-aus-144.reviz.ai2.in:5000,jupiter-cs-aus-143.reviz.ai2.in:5000,jupiter-cs-aus-208.reviz.ai2.in:5000,jupiter-cs-aus-210.reviz.ai2.in:5000,jupiter-cs-aus-190.reviz.ai2.in:5000,jupiter-cs-aus-154.reviz.ai2.in:5000,jupiter-cs-aus-145.reviz.ai2.in:5000

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "tmax-15k DPPO Qwen35 9b (prod recipe; 4-node; 64k; holmes/cu13/B300; LOSS-SCALE FIX 50f22169 A/B vs swerl_qwen35_9b_dppo_prod_4node_64k_holmes; relaunch #4 (prev 01M26TTN/01M27D95/01M283KQ died to holmes gang churn) w/ RAY OTel off; RESUME from global_step36)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents-holmes \
       --priority urgent \
       --preemptible \
       --min_runtime 28800s \
       --auto_resume \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env RAY_enable_open_telemetry=0 \
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
       --env MIRROR_URL=$MIRRORS \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
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
    --tool_configs '{"task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/qwen35_9b_dppo_lossfix_holmes_001 \
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
