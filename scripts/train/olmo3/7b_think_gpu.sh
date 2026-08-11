#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8-GPU, 20-step Olmo-3-7B-Think RL smoke on Dolci math.
# Change MODEL_PATH and DATASET_PATH for another local model or dataset.
#
# Usage:
#   bash /home/z30077923/open-instruct/7b_think_gpu.sh

REPO_ROOT="/home/z30077923/open-instruct"
MODEL_PATH="/home/z30077923/Olmo-3-7B-Think-DPO"
DATASET_PATH="/home/z30077923/Dolci-Think-RL-7B"
OUTPUT_DIR="/home/z30077923/open-instruct/output/7b_think_gpu"
DATASET_CACHE_DIR="/home/z30077923/open-instruct/output/7b_think_gpu/dataset_cache"
MATH_DATASET_DIR="${OUTPUT_DIR}/dolci_math_64"
MATH_DATASET_FILE="${MATH_DATASET_DIR}/train.parquet"
LOG_FILE="${OUTPUT_DIR}/run.log"

for required_path in "${MODEL_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}" "${DATASET_CACHE_DIR}" "${MATH_DATASET_DIR}"
cd "${REPO_ROOT}"

if [[ ! -s "${MATH_DATASET_FILE}" ]]; then
    echo "Preparing the deterministic first 64 math prompts from the local Dolci dataset..."
    python - "${DATASET_PATH}" "${MATH_DATASET_FILE}" <<'PY'
import sys

from datasets import load_dataset

source_path, output_path = sys.argv[1:]
source = load_dataset(source_path, split="train", num_proc=8)
math = source.filter(lambda row: row["dataset"] == ["math"], num_proc=8)
if len(math) != 30180:
    raise RuntimeError(f"Expected 30180 Dolci math prompts, found {len(math)}")
math.select(range(64)).to_parquet(output_path)
print(f"saved {output_path} with the deterministic first 64 of {len(math)} math prompts")
PY
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ ]] || ((GPU_COUNT < 8)); then
    echo "This recipe requires at least 8 visible GPUs; detected ${GPU_COUNT}." >&2
    exit 2
fi

echo "model=${MODEL_PATH}"
echo "dataset=${MATH_DATASET_FILE} (deterministic first 64 Dolci math prompts)"
echo "output=${OUTPUT_DIR}"
echo "topology=6 DeepSpeed learners + 2 vLLM engines on one node"
echo "training_steps=20"
echo "response_length=6144 pack_length=8192"

python open_instruct/grpo_fast.py \
    --exp_name 7b_think_gpu \
    --output_dir "${OUTPUT_DIR}" \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 2 \
    --num_unique_prompts_rollout 12 \
    --num_mini_batches 2 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list "${MATH_DATASET_DIR}" 64 \
    --dataset_mixer_list_splits train \
    --dataset_transform_fn rlvr_pre_tokenized_v1 rlvr_max_length_filter_v1 \
    --dataset_cache_mode local \
    --dataset_local_cache_dir "${DATASET_CACHE_DIR}" \
    --max_prompt_token_length 2048 \
    --response_length 6144 \
    --pack_length 8192 \
    --model_name_or_path "${MODEL_PATH}" \
    --chat_template_name olmo_thinker \
    --non_stop_penalty False \
    --mask_truncated_completions False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --total_episodes 480 \
    --async_steps 1 \
    --deepspeed_stage 3 \
    --num_learners_per_node 6 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_enforce_eager \
    --vllm_enable_prefix_caching \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every -1 \
    --save_freq 20 \
    --checkpoint_state_freq -1 \
    --gradient_checkpointing \
    --no_filter_zero_std_samples \
    --no_enable_queue_dashboard \
    --no_push_to_hub \
    --no_try_auto_save_to_beaker \
    2>&1 | tee "${LOG_FILE}"

echo "7B_THINK_GPU=PASS"
