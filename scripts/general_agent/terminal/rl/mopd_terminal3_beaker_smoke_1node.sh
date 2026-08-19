#!/bin/bash

# Beaker smoke (stage 3 of terminal-MOPD): MULTI-TEACHER OPD from three
# TERMINAL EXPERT teachers, routed per domain:
#   termigen rollouts -> allenai/qwen35-9b-termigen
#   endless rollouts  -> allenai/qwen35-9b-endless
#   tmax rollouts     -> allenai/tmax-9b
# tmax-2b student, pure OPD, 1 node / 8 GPUs (4 ZeRO-3 learners — the three
# 9B teachers shard to ~13.5GB/GPU — + 4 engines), ~16 steps.
#
# The in-job prep step builds the merged task-data dir AND re-tagged jsonls
# (termigen/endless/tmax) — routing needs distinguishable dataset tags since
# all three ship as "passthrough". Re-tagged rows skip the env reward, which
# is fine under --opd_pure. Watch objective/opd_route_frac_teacher_{0,1,2}
# (all > 0 across steps) and per-teacher reverse KLs.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

EXP_NAME=${EXP_NAME:-mopd_terminal3_beaker_smoke_tmax2b}

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "Terminal-MOPD stage-3 smoke: tmax-2b <- 3 terminal 9B experts, routed (1 node)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
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
       --env MIRROR_URL=jupiter-cs-aus-106.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python scripts/data/prepare_terminal3_mopd.py --task-data-dir /tmp/terminal3_task_data --jsonl-dir /tmp/mopd_terminal3 --rows-per-dataset 512 \&\& python open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path allenai/tmax-2b \
    --dataset_mixer_list /tmp/mopd_terminal3/termigen.jsonl 1.0 /tmp/mopd_terminal3/endless.jsonl 1.0 /tmp/mopd_terminal3/tmax.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --opd_teacher_model_name_or_path allenai/qwen35-9b-termigen allenai/qwen35-9b-endless allenai/tmax-9b \
    --opd_teacher_combine route \
    --opd_teacher_domains termigen endless tmax \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 4096 \
    --response_length 16384 \
    --pack_length 18432 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 1024 \
    --deepspeed_stage 3 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --load_ref_policy false \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_dir": "/tmp/terminal3_task_data", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --pool_size 128 \
    --max_steps 32 \
    --backend_timeout 600 \
    --gradient_checkpointing \
    --local_eval_every -1 \
    --logging_steps 1 \
    --seed 42 \
    --with_tracking \
    --wandb_project oe-general-agents \
    --push_to_hub false
