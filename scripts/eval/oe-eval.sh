#!/bin/bash

set -ex

# A script for using oe-eval for our development!
# to use, clone oe-eval (https://github.com/allenai/oe-eval-internal) into the top level dir of this repo.
# you'll need the current main for this to work.
# sadly, this is internal at Ai2 for now, but we are working on making it public!

# Example usages:
# ./scripts/eval/oe-eval.sh --model-name <model_name>  --model-location <model_path> [--hf-upload] [--max-length <max_length>]
# model_name should be a human-readable name for the model/run. This will be used in experiment tracking.
# model_path should be
#   (a) a huggingface name (e.g. allenai/llama-3-tulu-2-8b),
#   (b) a beaker dataset name (e.g. beaker://hamishivi/olmo_17_7b_turbo_dpo) - note the beaker://
#   (c) a beaker dataset hash (e.g., beaker://01J28FDK3GDNA2C5E9JXBW1TP4) - note the beaker://
#   (d) (untested) an absolute path to a model on cirrascale nfs.
# hf-upload is an optional flag to upload the results to huggingface for result tracking.
# e.g.:
# ./scripts/eval/oe-eval.sh --model-name olmo_17_7b_turbo_sft  --model-location beaker://01J28FDK3GDNA2C5E9JXBW1TP4 --hf-upload --max-length 2048
# ./scripts/eval/oe-eval.sh --model-name llama-3-tulu-2-dpo-8b --model-location allenai/llama-3-tulu-2-8b --hf-upload
# Specifying unseen-evals evaluates the model on the unseen evaluation suite instead of the development suite.

# Tulu eval dev suite is:
# gsm8k::tulu
# bbh:cot::tulu
# drop::llama3
# minerva_math::tulu
# codex_humaneval::tulu
# codex_humanevalplus::tulu
# ifeval::tulu
# popqa::tulu
# mmlu:mc::tulu
# alpaca_eval_v2::tulu
# truthfulqa::tulu

# Tulu eval unseen suite is:
# agi_eval_english:0shot_cot::tulu3
# gpqa:0shot_cot::tulu3
# mmlu_pro:0shot_cot::tulu3
# deepmind_math:0shot_cot::tulu3
# bigcodebench_hard::tulu
# gpqa:0shot_cot::llama3.1
# mmlu_pro:cot::llama3.1
# bigcodebench::tulu


# Function to print usage
usage() {
    echo "Usage: $0 --model-name MODEL_NAME --model-location MODEL_LOCATION [--num_gpus GPUS] [--hf-upload] [--revision REVISION] [--max-length <max_length>] [--unseen-evals] [--priority priority] [--tasks TASKS] [--evaluate_on_weka]"
    echo "TASKS should be a comma-separated list of task specifications (e.g., 'gsm8k::tulu,bbh:cot::tulu')"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift ;;
        --model-location) MODEL_LOCATION="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        --hf-upload) HF_UPLOAD="true" ;;
        --revision) REVISION="$2"; shift ;;
        --max-length) MAX_LENGTH="$2"; shift ;;
        --unseen-evals) UNSEEN_EVALS="true" ;;
        --priority) PRIORITY="$2"; shift ;;
        --tasks) CUSTOM_TASKS="$2"; shift ;;
        --evaluate_on_weka) EVALUATE_ON_WEKA="true" ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Optional: Default number of GPUs if not specified
NUM_GPUS="${NUM_GPUS:-1}"

# Check required arguments
if [[ -z "$MODEL_NAME" || -z "$MODEL_LOCATION" ]]; then
    echo "Error: --model-name and --model-location are required."
    usage
fi

# Replace '/' with '_' in MODEL_NAME
MODEL_NAME_SAFE=${MODEL_NAME//\//_}

# Set defaults for optional arguments
HF_UPLOAD="${HF_UPLOAD:-false}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
UNSEEN_EVALS="${UNSEEN_EVALS:-false}"
PRIORITY="${PRIORITY:normal}"
EVALUATE_ON_WEKA="${EVALUATE_ON_WEKA:-false}"

# Set HF_UPLOAD_ARG if HF_UPLOAD is true
if [ "$HF_UPLOAD" == "true" ]; then
    # if UNSEEN_EVALS, save results to a different directory
    if [ "$UNSEEN_EVALS" == "true" ]; then
        HF_UPLOAD_ARG="--hf-save-dir allenai/tulu-3-evals-unseen//results/${MODEL_NAME_SAFE}"
    else
        HF_UPLOAD_ARG="--hf-save-dir allenai/tulu-3-evals//results/${MODEL_NAME_SAFE}"
    fi
else
    HF_UPLOAD_ARG=""
fi

# Set REVISION if not provided
if [[ -n "$REVISION" ]]; then
    REVISION_ARG="--revision $REVISION"
else
    REVISION_ARG=""
fi

# Define default tasks if no custom tasks provided
DEFAULT_TASKS=(
    "gsm8k::tulu"
    "bbh:cot::tulu"
    "drop::llama3"
    "minerva_math::tulu"
    "codex_humaneval::tulu"
    "codex_humanevalplus::tulu"
    "ifeval::tulu"
    "popqa::tulu"
    "mmlu:mc::tulu"
    "mmlu:cot::summarize"
    "alpaca_eval_v2::tulu"
    "truthfulqa::tulu"
)
UNSEEN_TASKS=(
    "agi_eval_english:0shot_cot::tulu3"
    "gpqa:0shot_cot::tulu3"
    "mmlu_pro:0shot_cot::tulu3"
    "deepmind_math:0shot_cot-v3::tulu3"
    "bigcodebench_hard::tulu"
    "gpqa:0shot_cot::llama3.1"
    "mmlu_pro:cot::llama3.1"
    "bigcodebench::tulu"
)

# If custom tasks provided, convert comma-separated string to array
if [ "$UNSEEN_EVALS" == "true" ]; then
    TASKS=("${UNSEEN_TASKS[@]}")
elif [[ -n "$CUSTOM_TASKS" ]]; then
    IFS=',' read -ra TASKS <<< "$CUSTOM_TASKS"
else
    TASKS=("${DEFAULT_TASKS[@]}")
fi

MODEL_TYPE="--model-type vllm"
BATCH_SIZE_VLLM=10000
BATCH_SIZE_OTHER=1
# Set GPU_COUNT and GPU_COUNT_OTHER based on NUM_GPUS
GPU_COUNT="$NUM_GPUS"
GPU_COUNT_OTHER=$((NUM_GPUS * 2))
MODEL_TYPE_OTHER=""

for TASK in "${TASKS[@]}"; do
    # mmlu and truthfulqa need different batch sizes and gpu counts because they are multiple choice and we cannot use vllm.
    if [[ "$TASK" == "mmlu:mc::tulu" || "$TASK" == "truthfulqa::tulu" ]]; then
        BATCH_SIZE=$BATCH_SIZE_OTHER
        GPU_COUNT=$GPU_COUNT_OTHER
        MODEL_TYPE=$MODEL_TYPE_OTHER
    else
        BATCH_SIZE=$BATCH_SIZE_VLLM
        MODEL_TYPE="--model-type vllm"
        GPU_COUNT=$GPU_COUNT
    fi

    if [ "$EVALUATE_ON_WEKA" == "true" ]; then
        python oe-eval-internal/oe_eval/launch.py \
            --model "$MODEL_NAME" \
            --beaker-workspace "ai2/tulu-3-results" \
            --beaker-budget ai2/oe-adapt \
            --task "$TASK" \
            $MODEL_TYPE \
            --batch-size "$BATCH_SIZE" \
            --model-args "{\"model_path\":\"${MODEL_LOCATION}\", \"max_length\": ${MAX_LENGTH}}" \
            ${HF_UPLOAD_ARG} \
            --gpus "$GPU_COUNT" \
            --beaker-image "costah/oe-eval-olmo1124-11142024" \
            --gantry-args '{"env-secret": "OPENAI_API_KEY=openai_api_key", "weka": "oe-adapt-default:/weka/oe-adapt-default"}' \
            ${REVISION_ARG} \
            --cluster ai2/neptune-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2 \
            --beaker-retries 2 \
            --beaker-priority "$PRIORITY"
    else
        python oe-eval-internal/oe_eval/launch.py \
        --model "$MODEL_NAME" \
        --beaker-workspace "ai2/tulu-3-results" \
        --beaker-budget ai2/oe-adapt \
        --task "$TASK" \
        $MODEL_TYPE \
        --batch-size "$BATCH_SIZE" \
        --model-args "{\"model_path\":\"${MODEL_LOCATION}\", \"max_length\": ${MAX_LENGTH}}" \
        ${HF_UPLOAD_ARG} \
        --gpus "$GPU_COUNT" \
        --beaker-image "costah/oe-eval-olmo1124-11142024" \
        --gantry-args '{"env-secret": "OPENAI_API_KEY=openai_api_key"}' \
        ${REVISION_ARG} \
        --beaker-retries 2 \
        --beaker-priority "$PRIORITY"
    fi
done
