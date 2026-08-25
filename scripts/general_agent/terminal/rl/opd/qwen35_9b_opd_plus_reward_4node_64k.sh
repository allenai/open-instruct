#!/bin/bash

# COMBINED objective: OPD distillation + the binary terminal (verifier) reward.
# Identical to qwen35_9b_dppo_opd_4node_64k.sh except `--opd_pure` is REMOVED,
# so advantages are  A_t = A_env,t - opd_kl_coef * reverse_kl_t  instead of the
# distillation term alone.
#
# Motivation: pure OPD provably cannot EXCEED its teacher (reverse KL has no
# signal beyond matching it) — the 9B pure run confirmed this, plateauing at
# teacher level (TB2.1 0.28-0.29 vs teacher 0.276; TBlite 0.54-0.58 vs 0.534)
# while the DPPO RL run eventually reached 0.301/0.582. Adding the reward gives
# the student a reason to go past the teacher.
#
# Why the two terms complement rather than duplicate:
#   * On prompt groups where all rollouts succeed or all fail, the centered
#     reward advantage is exactly 0 (nothing for RL to learn) but the KL term is
#     not — so OPD teaches on the groups RL discards. Hence
#     `--filter_zero_std_samples false` and NO `--active_sampling`: keep those
#     groups instead of filtering them away.
#   * The KL term starts ~0.22 (comparable to centered reward advantages) and
#     decays to ~0.01 as the student converges, so the objective auto-anneals
#     from distillation-dominant to reward-dominant with no schedule.
#
# opd_kl_coef stays 1.0 for comparability with every other run in the campaign.
# Baselines to beat: pure OPD 0.292/0.580 (best), teacher 0.276/0.534,
# DPPO-best 0.301/0.582 (step ~360, days of training), base 0.193/0.432.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B
TEACHER_MODEL=allenai/tmax-9b

EXP_NAME=swerl_qwen35_9b_opd_plus_reward_4node_64k

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "OPD + binary terminal reward (combined objective): base Qwen3.5-9B <- tmax-9b; 4-node; 64k" \
       --pure_docker_mode \
       --workspace ai2/general-tool-use \
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
       --env MIRROR_URL=jupiter-cs-aus-106.reviz.ai2.in:5000 \
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
    --save_trainer_logprobs true \
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
    --try_launch_beaker_eval_jobs_on_weka False
