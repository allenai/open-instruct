#!/bin/bash

# PRODUCTION terminal-expert MOPD: distill THREE terminal 9B experts into base
# Qwen3.5-9B on the students' own rollouts, routed per domain (MOPD,
# arXiv:2606.30406 — same-origin teachers, per the paper's key finding):
#   termigen rollouts -> allenai/qwen35-9b-termigen
#   endless rollouts  -> allenai/qwen35-9b-endless
#   tmax rollouts     -> allenai/tmax-9b
#
# 4-node twin of qwen35_9b_dppo_opd_4node_64k.sh (the validated single-teacher
# 9B OPD prod recipe: 16 learners SP=4 + 16 engines, 64k, pure OPD). The three
# teachers share the student's attention geometry, so all use the cheap
# sharded scoring path (~3 extra no-grad forwards/step; runs are
# rollout-bound). Dataset shatu/terminal3-mopd is the self-contained merged
# repo: re-tagged termigen+endless+tmax rows (routing needs distinct tags;
# passthrough env reward is skipped, harmless under --opd_pure) + one merged
# task-data.tar.gz, so every env actor on every node bootstraps via the
# standard task_data_hf_repo flow.
#
# Watch objective/opd_reverse_kl (should FALL; local/1-node smokes: 0.56->0.25
# in 16 steps at 2B), per-teacher opd_reverse_kl_teacher_{0,1,2}, and
# opd_route_frac_teacher_{0,1,2} (should track the ~24%/17%/59% dataset mix).
# Smoke pedigree: local 4xL40S + 1-node Beaker (01KZZ8MQZF280FAKY5ZYFPFT5E).

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

EXP_NAME=swerl_mopd_terminal3_9b_4node_64k

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "Terminal-expert MOPD prod: Qwen3.5-9B <- termigen/endless/tmax 9B experts, routed (4-node; 64k)" \
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
    --dataset_mixer_list shatu/terminal3-mopd 1.0 \
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
    --opd_teacher_model_name_or_path allenai/qwen35-9b-termigen allenai/qwen35-9b-endless allenai/tmax-9b \
    --opd_teacher_combine route \
    --opd_teacher_domains termigen endless tmax \
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
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "shatu/terminal3-mopd", "test_timeout": 120, "image": "python:3.12-slim"}' \
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
