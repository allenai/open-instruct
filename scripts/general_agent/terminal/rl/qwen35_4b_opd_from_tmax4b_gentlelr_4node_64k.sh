#!/bin/bash

# OPD "gentle optimization" arm — identical to
# qwen35_4b_opd_from_tmax4b_4node_64k.sh except: lr 1e-6 -> 5e-7 and
# constant -> constant_with_warmup (warmup_ratio 0.05 ~= 25 steps of the ~500).
#
# Hypothesis under test: the early eval dip in the same-lineage OPD runs
# (base 4B TB2.1 0.135 -> 0.101 at step 20 before recovering to 0.202 at s40)
# is optimization damage, not distillation. In pure OPD the per-token advantage
# IS the raw reverse KL (unnormalized), so updates are LARGEST at step 1
# (KL 0.65 here) and decay 10-30x — the opposite of RL's stationary advantage
# scale, and exactly when Adam's moments are coldest. Production DPPO/tmax
# recipes all use constant 1e-6 with no warmup; this arm deliberately breaks
# that inheritance. NOTE: --opd_kl_coef would NOT test this (AdamW normalizes a
# uniform loss rescale away); LR/warmup are the knobs that change step size.
#
# Union test on purpose (both knobs at once) to maximize the chance of seeing an
# effect on a GPU budget; decompose only if the dip disappears.
# Baseline to compare against: swerl_qwen35_4b_opd_from_tmax4b_4node_64k
# (TB2.1 s20 0.101 / s40 0.202 / s60 0.112 / s80 0.180 / s100 0.135).
#
# Released 4B recipe otherwise: 4 nodes = 16 learners (8 8, SP=4) + 16 engines,
# 64k. Pure OPD: rewards logged, no gradient. Watch objective/opd_reverse_kl.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=hamishivi/Qwen3.5-4B
TOKENIZER=hamishivi/Qwen3.5-4B
TEACHER_MODEL=allenai/tmax-4b

EXP_NAME=swerl_qwen35_4b_opd_from_tmax4b_gentlelr_4node_64k

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "OPD gentle-LR arm: base Qwen3.5-4B <- tmax-4b, lr 5e-7 + 5% warmup (dip hypothesis test; 4-node; 64k)" \
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
    --opd_pure \
    --filter_zero_std_samples false \
    --temperature 1.0 \
    --learning_rate 5e-7 \
    --total_episodes 128000 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
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
    --try_launch_beaker_eval_jobs_on_weka False
