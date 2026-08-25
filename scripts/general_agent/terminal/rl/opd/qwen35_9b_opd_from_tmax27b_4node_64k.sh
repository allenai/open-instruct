#!/bin/bash

# OPD: base Qwen3.5-9B student <- released tmax-27b teacher.
#
# Fills the hole in the capacity-gap story. Every large-gap run so far used a
# 4B student, so "big teacher/student gap hurts" and "a 4B student can't absorb
# a big teacher" were confounded. This run holds the student fixed at base 9B
# (identical to swerl_qwen35_9b_dppo_opd_4node_64k) and swaps ONLY the teacher
# 9b -> 27b, i.e. a 3x gap between the two we already measured:
#     base 9B <- tmax-9b   1.0x   KL floor ~0.01, no trough
#     base 9B <- tmax-27b  3.0x   <- this run
#     tmax-4b <- tmax-27b  6.75x  KL floor ~0.17, deep trough, stopped at ~120
#
# Teacher = allenai/tmax-27b: same qwen3_5 architecture family, byte-identical
# tokenizer (verified 2026-08-03), loaded learner-side and ZeRO-3-sharded over
# the 16 learner GPUs (~3.4 GB/GPU of weights vs ~1.1 GB/GPU for the 9B
# teacher). Its attention geometry differs from the 9B student's, so under SP
# the teacher scores the full pre-split sequences with tiled lm-head logprobs
# (auto-detected in grpo_fast; needs commit 42e9cc660's late attention-registry
# resolution). That full-sequence path is the main memory risk here: if this
# OOMs where the 9B<-9b arm did not, drop --num_samples_per_prompt_rollout to
# 16 before touching pack_length.
#
# LR is 1e-6, NOT the 5e-7 we now prefer, so the only difference vs the 9B<-9b
# reference arm is the teacher. If a destructive trough appears, re-run at
# 5e-7 to separate "gap too big" from "steps too big".
#
# Pure OPD: verifier rewards are logged but carry no gradient; reward-variance
# filtering disabled (no --active_sampling). Watch objective/opd_reverse_kl
# (should fall toward a floor set by the capacity ratio) and scores.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B
TEACHER_MODEL=allenai/tmax-27b

EXP_NAME=swerl_qwen35_9b_opd_from_tmax27b_4node_64k

# Resuming after Beaker exhausts --max_retries. mason.py normally stamps a
# fresh --checkpoint_state_dir per submit (which would silently restart from
# step 0), but maybe_override_checkpoint_dir() leaves a caller-supplied path
# alone when it is on /weka -- so passing it explicitly is what makes a manual
# relaunch continue rather than start over.
RESUME_ARGS=()
RESUME_NOTE=""
if [[ -n "${OPD_RESUME_STATE_DIR:-}" ]]; then
    RESUME_ARGS=(--checkpoint_state_dir "$OPD_RESUME_STATE_DIR")
    RESUME_STEP=$(cat "$OPD_RESUME_STATE_DIR/latest" 2>/dev/null || echo unknown)
    RESUME_NOTE=" [resumed from $RESUME_STEP]"
fi

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "OPD: base Qwen3.5-9B student <- tmax-27b teacher (pure distill; 3x capacity gap; 4-node; 64k)${RESUME_NOTE}" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
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
       --env MIRROR_URL=jupiter-cs-aus-208.reviz.ai2.in:5000 \
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
    --opd_teacher_model_name_or_path $TEACHER_MODEL \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
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
    --try_launch_beaker_eval_jobs_on_weka False \
    "${RESUME_ARGS[@]}"
