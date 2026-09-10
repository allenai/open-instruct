#!/bin/bash

# OPD arm B of the open-instruct-vs-slime comparison, on cu13/B300/holmes.
# = arm A (--async_steps 1; jupiter run 01M244GSZXDJ552JYMWJ9E8WBW) PLUS
#   --opd_student_logprobs learner: the student side of the reverse-KL advantage
#   is a detached forward of the CURRENT trainer policy (same tiled path +
#   lm_head_fp32 as the teacher) instead of the vLLM rollout logprobs. This is
#   slime's OPD signal exactly (use_rollout_logprobs=False): no staleness, no
#   vLLM-vs-HF numerics in the advantage. Costs one extra no-grad 4B forward
#   per step (~+30% time/training). Watch debug/opd_student_learner_vs_vllm_
#   logprob_diff_mean for how far the two student estimates actually are.
# Everything else identical to the vjgol9zb baseline (DPPO tv/0.1 kept, so the
# surrogate ratio is still anchored on vLLM; only the KL's student side moves).
#
# Holmes/cu13 deltas (from qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh):
# cluster ai2/holmes, workspace ai2/oe-agents-holmes, --min_runtime 8h +
# --auto_resume, --attn_implementation flash_4, triton GDN prefill, explicit
# checkpoint_state_dir with state_freq 5. Image: branch opd_cuda13.
#
#   ./scripts/train/build_image_and_launch_dirty.sh --cuda-version 13 \
#       scripts/general_agent/terminal/rl/opd/qwen35_4b_opd_from_tmax9b_4node_64k_async1_learnerkl_holmes_cu13.sh
#   # or, image already built:
#   bash scripts/general_agent/terminal/rl/opd/qwen35_4b_opd_from_tmax9b_4node_64k_async1_learnerkl_holmes_cu13.sh \
#       shashankg/open-instruct-integration-test-opd_cuda13-cuda13
#
# MIRROR_URL is a comma-separated list (tried in order); all three verified live
# + warmed 2026-09-09 (registry-mirror-oe-agents-{1,2,3}-20260909). Jupiter
# mirrors are reachable from holmes.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-4B
TOKENIZER=hamishivi/Qwen3.5-4B
TEACHER_MODEL=allenai/tmax-9b

EXP_NAME=swerl_qwen35_4b_opd_from_tmax9b_4node_64k_async1_learnerkl_holmes

uv run python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "OPD arm B: async_steps 1 + learner-side student logprobs in the reverse KL (slime-matched; baseline vjgol9zb = async 4 + vLLM student) -- base Qwen3.5-4B <- tmax-9b pure distill; holmes/cu13/B300; 4-node 64k" \
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
       --env MIRROR_URL=jupiter-cs-aus-145.reviz.ai2.in:5000,jupiter-cs-aus-144.reviz.ai2.in:5000,jupiter-cs-aus-143.reviz.ai2.in:5000 \
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
    --async_steps 1 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
    --opd_teacher_model_name_or_path $TEACHER_MODEL \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --opd_student_logprobs learner \
    --filter_zero_std_samples false \
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
    --save_trainer_logprobs false \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/qwen35_4b_opd_tmax9b_async1_learnerkl_holmes_001 \
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
