#!/bin/bash
# Variant of recreate_blowup_mask.sh that doubles the generation budget to 64k
# and moves sequence parallel from 2 -> 4 so per-GPU activation memory stays flat.
# 4 nodes x 8 GPUs (32 GPUs total); DP per node = 8 / sp=4 = 2.
#
# Response + prompt fit:
#   max_prompt_token_length 2048 + response_length 65536 = 67584 min.
#   pack_length 68608 keeps the same ~1k slack that 35840 had over 34816.
#
# Mask mapping (unchanged from recreate_blowup_mask.sh):
#   --tis_mask_lower 0.5 -> ratio lower bound = 0.5
#   --tis_mask_upper 2.0 -> ratio upper bound = 3.0
#   --clip_higher    2.0 -> keeps CISPO's internal clamp at the mask upper.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "SWERL tmax-10k GRPO with Qwen3.5-9B blowup repro + CISPO mask, 64k ctx, sp=4" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 4 \
       --max_retries 0 \
       --env REPO_PATH=/stage \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=hamishi740 \
       --secret DOCKER_PAT=hamishivi_DOCKER_PAT \
       --budget ai2/oe-adapt \
       --mount_docker_socket \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 65536 \
    --pack_length 68608 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path hamishivi/qwen3.5_tmax_breakdown_test_step100 \
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
    --loss_fn cispo \
    --clip_higher 2.0 \
    --tis_mask_lower 0.5 \
    --tis_mask_upper 2.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enforce_eager \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 100 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name swerl_qwen35_9b_base_tmax_10k_grpo_breakdown_mask_64k \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
