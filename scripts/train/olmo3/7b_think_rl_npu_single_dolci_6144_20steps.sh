#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8-NPU, 20-step Olmo-3-7B-Think RL run on the same
# deterministic first 64 math prompts from the complete local Dolci dataset.
# Compared with the original run, response_length is increased from 2048 to
# 6144 and pack_length from 4096 to 8192. No synthetic reward is injected.
#
# Launch with:
#   bash --noprofile --norc -ic 'source ~/.bashrc && bash scripts/train/olmo3/7b_think_rl_npu_single_dolci_6144_20steps.sh'

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is empty. Launch through the interactive command in this file." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_ROOT="/data/zyh/miniconda3/envs/open_instruct_npu_py312"
PYTHON_BIN="${ENV_ROOT}/bin/python"
MODEL_PATH="/data/model/Olmo-3-7B-Think-DPO"
DATASET_PATH="/data/dataset/Dolci-Think-RL-7B"
OUTPUT_DIR="${REPO_ROOT}/output/olmo3_7b_think_rl_npu_single_dolci_math64_20steps_r6144"
DATASET_CACHE_DIR="${OUTPUT_DIR}/dataset_cache"
MATH_DATASET_DIR="${OUTPUT_DIR}/dolci_math_64"
MATH_DATASET_FILE="${MATH_DATASET_DIR}/train.parquet"
LOG_FILE="${OUTPUT_DIR}/run.log"
RAY_TMPDIR="/data/zyh/raytmp"
TMPDIR="/data/zyh/oi_rl_tmp"

for required_path in "${ENV_ROOT}" "${MODEL_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/data/zyh/cache/open_instruct_hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export RAY_TMPDIR
export TMPDIR

mkdir -p "${OUTPUT_DIR}" "${DATASET_CACHE_DIR}" "${MATH_DATASET_DIR}" "${RAY_TMPDIR}" "${TMPDIR}"
cd "${REPO_ROOT}"

if [[ ! -s "${MATH_DATASET_FILE}" ]]; then
    echo "Preparing the deterministic first 64 math prompts from the complete local Dolci dataset..."
    "${PYTHON_BIN}" - "${DATASET_PATH}" "${MATH_DATASET_FILE}" <<'PY'
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

NPU_COUNT="$("${PYTHON_BIN}" -c 'import torch, torch_npu; assert torch.npu.is_available(); print(torch.npu.device_count())')"
if [[ ! "${NPU_COUNT}" =~ ^[0-9]+$ ]] || ((NPU_COUNT < 8)); then
    echo "This run requires 8 visible NPUs; detected ${NPU_COUNT}." >&2
    exit 2
fi

echo "model=${MODEL_PATH}"
echo "dataset=${MATH_DATASET_FILE} (same first 64 math prompts from ${DATASET_PATH})"
echo "output=${OUTPUT_DIR}"
echo "topology=6 DeepSpeed learners + 2 vLLM-Ascend engines on one node"
echo "training_steps=20"
echo "response_length=6144 pack_length=8192"
echo "vllm=2xTP1 memory_utilization=0.7 compile=enabled prefix_caching=enabled"
echo "learner_attention=sdpa reference_policy=disabled(beta=0)"

set +e
timeout 28800 "${PYTHON_BIN}" open_instruct/grpo_fast.py \
    --exp_name olmo3_7b_think_rl_npu_single_dolci_math64_20steps_r6144 \
    --output_dir "${OUTPUT_DIR}" \
    --beta 0.0 \
    --no_load_ref_policy \
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
    --vllm_gpu_memory_utilization 0.7 \
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
pipeline_status=("${PIPESTATUS[@]}")
set -e
status="${pipeline_status[0]}"
if [[ "${status}" != 0 ]]; then
    echo "RL command failed with exit code ${status}; inspect ${LOG_FILE}." >&2
    exit "${status}"
fi

if ! grep -Eq "training_step: 20" "${LOG_FILE}"; then
    echo "RL log does not contain the twentieth training step: ${LOG_FILE}" >&2
    exit 1
fi

echo "RUNTIME_SELECTED_DEVICE=npu device=npu"
echo "OLMO3_7B_THINK_RL_NPU_DOLCI_6144_20STEPS=PASS"
