#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-math-rlvr-smoke}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"
BEAKER_USER="$(beaker account whoami --format json | jq -r '.[0].name')"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/open-instruct-dev}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-30B-A3B}"
DATASET="${DATASET:-ai2-adapt-dev/rlvr_gsm8k_zs}"
OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-adapt-default/${BEAKER_USER}/qmoe-int/qwen3-30b-a3b-math-rlvr}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-5}"
NUM_NODES="${NUM_NODES:-1}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-4}"
OLMO_CORE_EP_DEGREE="${OLMO_CORE_EP_DEGREE:-4}"
ROUTER_REPLAY="${ROUTER_REPLAY:-false}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
OLMO_USE_TORCH_GROUPED_MM="${OLMO_USE_TORCH_GROUPED_MM:-true}"
OLMO_CORE_CUDA_STAGE_SYNC="${OLMO_CORE_CUDA_STAGE_SYNC:-0}"
RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
TOTAL_EPISODES=$((NUM_UNIQUE_PROMPTS * NUM_SAMPLES_PER_PROMPT * NUM_TRAINING_STEPS))

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace "$BEAKER_WORKSPACE" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --preemptible \
    --no_auto_dataset_cache \
    --num_nodes "$NUM_NODES" \
    --gpus 8 \
    --max_retries 0 \
    --timeout 6h \
    --artifact_ttl 3d \
    --env OLMO_SHARED_FS=1 \
    --env CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
    --env OLMO_USE_TORCH_GROUPED_MM="$OLMO_USE_TORCH_GROUPED_MM" \
    --env OLMO_CORE_CUDA_STAGE_SYNC="$OLMO_CORE_CUDA_STAGE_SYNC" \
    --env RAY_DEDUP_LOGS="$RAY_DEDUP_LOGS" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -- source configs/beaker_configs/ray_node_setup.sh \
    \&\& python open_instruct/grpo.py \
        --exp_name "$EXP_NAME" \
        --run_name "$RUN_NAME" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --attn_implementation flash_4 \
        --compile_model false \
        --dataset_mixer_list "$DATASET" 64 \
        --dataset_mixer_list_splits train \
        --max_prompt_token_length 256 \
        --response_length 1024 \
        --pack_length 2048 \
        --num_unique_prompts_rollout "$NUM_UNIQUE_PROMPTS" \
        --num_samples_per_prompt_rollout "$NUM_SAMPLES_PER_PROMPT" \
        --async_steps 1 \
        --filter_zero_std_samples false \
        --debug_grpo_diagnostics true \
        --per_device_train_batch_size 1 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --total_episodes "$TOTAL_EPISODES" \
        --learning_rate 1e-6 \
        --lr_scheduler_type constant \
        --temperature 1.0 \
        --non_stop_penalty false \
        --mask_truncated_completions false \
        --apply_verifiable_reward true \
        --ground_truths_key ground_truth \
        --olmo_core_train_module ddp \
        --olmo_core_ep_degree "$OLMO_CORE_EP_DEGREE" \
        --router_replay "$ROUTER_REPLAY" \
        --num_learners_per_node "$NUM_LEARNERS_PER_NODE" \
        --vllm_num_engines 1 \
        --vllm_tensor_parallel_size 1 \
        --vllm_gpu_memory_utilization 0.5 \
        --vllm_enforce_eager \
        --vllm_sync_backend nccl \
        --inflight_updates false \
        --use_rho_correction true \
        --beta 0.0 \
        --load_ref_policy false \
        --local_eval_every -1 \
        --save_freq 1000 \
        --checkpoint_state_freq 1000 \
        --output_dir "$OUTPUT_DIR" \
        --try_auto_save_to_beaker false \
        --push_to_hub false \
        --save_traces \
        --verbose
