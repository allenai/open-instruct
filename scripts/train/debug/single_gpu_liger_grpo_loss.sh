#!/bin/bash

# Single-GPU GRPO smoke test for the tiled lm-head loss (`--use_liger_grpo_loss`).
#
# Mirrors single_gpu_on_beaker.sh, but runs under ZeRO-3 with the tiled loss
# enabled. ZeRO-3 is the point: the tiled kernel flips `param.ds_grad_is_ready`
# per tile so the lm-head gradient is reduced once per step rather than once per
# tile, and that interaction only exists under stage 3. `beta` is non-zero so the
# reference-KL branch of the kernel is exercised too.
#
# Runs preemptible: this is a short verification job, and a non-preemptible urgent
# request carries an 8-hour minRuntime guarantee that left it unscheduled for a day.
# Keep single_gpu_on_beaker.sh on the default configuration — it is the canonical
# single-GPU test — and use this script for the opt-in loss path.

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "Single GPU tiled (liger) GRPO lm-head loss test script." \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 15m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --gpus 1 \
       --no_auto_dataset_cache \
       --artifact_ttl 1d \
	   -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --add_bos \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --inflight_updates True \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 3 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --load_ref_policy true \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode
