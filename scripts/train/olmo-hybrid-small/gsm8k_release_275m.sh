#!/bin/bash
set -euo pipefail

# Pass this script to build_image_and_launch.sh; it supplies the committed image.
# 1,000 updates, 4 prompts x 4 completions, 1 learner + 1 vLLM engine.
# Beaker restarts load the latest complete native checkpoint from the stable Weka path.
BEAKER_IMAGE="${1:?Expected Beaker image from build_image_and_launch.sh}"
shift
RUN_NAME="${RUN_NAME:-hybrid-release-275m-gsm8k-1k-$(date -u +%Y%m%d-%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/275m/midtrain/step0038147}"
OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/$RUN_NAME}"

uv run --frozen python mason.py \
    --task_name "$RUN_NAME" \
    --description "Olmo Hybrid release 275m GSM8K: 1000 updates; native checkpoint/resume every 100 updates" \
    --cluster ai2/saturn \
    --workspace ai2/open-instruct-dev --budget ai2/oe-other \
    --priority normal --preemptible --max_retries 3 --timeout 24h \
    --pure_docker_mode --image "$BEAKER_IMAGE" --num_nodes 1 --gpus 2 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name "$RUN_NAME" --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" --checkpoint_state_dir "$OUTPUT_DIR/native" \
    --model_name_or_path "$MODEL_PATH" --model_dtype bfloat16 \
    --attn_implementation flash_2 --chat_template_name gsm8k_base \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list mnoukhov/gsm8k-platinum-openinstruct 1.0 \
    --dataset_mixer_eval_list_splits test \
    --beta 0.0 --load_ref_policy False --learning_rate 1e-6 \
    --lr_scheduler_type constant --num_epochs 1 --num_mini_batches 1 \
    --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 4 \
    --total_episodes 16000 --per_device_train_batch_size 1 \
    --async_steps 1 --inflight_updates False --active_sampling False \
    --filter_zero_std_samples False --advantage_normalization_type centered \
    --max_prompt_token_length 512 --response_length 2048 --pack_length 4096 \
    --fsdp_shard_degree 1 --fsdp_num_replicas 1 --num_learners_per_node 1 \
    --vllm_num_engines 1 --vllm_tensor_parallel_size 1 --vllm_enforce_eager \
    --vllm_attention_backend FLASH_ATTN --vllm_mamba_ssm_cache_dtype float32 \
    --vllm_gpu_memory_utilization 0.6 \
    --apply_verifiable_reward true --non_stop_penalty False --temperature 1.0 \
    --activation_checkpointing_mode selected_modules \
    --local_eval_every 100 --eval_on_step_0 --save_freq 100 --checkpoint_state_freq 100 \
    --seed 1 --with_tracking --wandb_entity mnoukhov --push_to_hub False "$@"
