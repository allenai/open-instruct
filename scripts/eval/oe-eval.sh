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
    echo "Usage: $0 --model-name MODEL_NAME --model-location MODEL_LOCATION [--num_gpus GPUS] [--upload_to_hf] [--revision REVISION] [--max-length <max_length>] [--task-suite TASK_SUITE] [--priority priority] [--tasks TASKS] [--evaluate_on_weka] [--stop-sequences <comma_separated_stops>] [--beaker-image <beaker_image>] [--cluster <clusters>] [--process-output <process_output>]"
    echo "TASK_SUITE should be one of: NEXT_MODEL_DEV, NEXT_MODEL_UNSEEN, TULU_3_DEV, TULU_3_UNSEEN (default: NEXT_MODEL_DEV)"
    echo "TASKS should be a comma-separated list of task specifications (e.g., 'gsm8k::tulu,bbh:cot::tulu')"
    echo "STOP_SEQUENCES should be a comma-separated list of strings to stop generation at (e.g., '</answer>,\\n\\n')"
    echo "PROCESS_OUTPUT should be a string specifying how to process the model output (e.g., 'r1_style')"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift ;;
        --model-location) MODEL_LOCATION="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        --upload_to_hf) UPLOAD_TO_HF="$2"; shift ;;
        --revision) REVISION="$2"; shift ;;
        --max-length) MAX_LENGTH="$2"; shift ;;
        --task-suite) TASK_SUITE="$2"; shift ;;
        --priority) PRIORITY="$2"; shift ;;
        --tasks) CUSTOM_TASKS="$2"; shift ;;
        --evaluate_on_weka) EVALUATE_ON_WEKA="true" ;;
        --step) STEP="$2"; shift ;;
        --run-id) RUN_ID="$2"; shift ;;
        --stop-sequences) STOP_SEQUENCES="$2"; shift ;;
        --beaker-image) BEAKER_IMAGE="$2"; shift ;;
        --cluster) CLUSTER="$2"; shift ;;
        --process-output) PROCESS_OUTPUT="$2"; shift ;;
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
MAX_LENGTH="${MAX_LENGTH:-4096}"
TASK_SUITE="${TASK_SUITE:-NEXT_MODEL_DEV}"
PRIORITY="${PRIORITY:normal}"
EVALUATE_ON_WEKA="${EVALUATE_ON_WEKA:-false}"
RUN_ID="${RUN_ID:-}"
STEP="${STEP:-}"

# Process stop sequences if provided
STOP_SEQUENCES_JSON=""
if [[ -n "$STOP_SEQUENCES" ]]; then
    # Convert comma-separated list to JSON array
    IFS=',' read -ra STOP_SEQS <<< "$STOP_SEQUENCES"
    STOP_SEQUENCES_JSON=", \"stop_sequences\": ["
    for i in "${!STOP_SEQS[@]}"; do
        if [ $i -gt 0 ]; then
            STOP_SEQUENCES_JSON+=", "
        fi
        STOP_SEQUENCES_JSON+="\"${STOP_SEQS[$i]}\""
    done
    STOP_SEQUENCES_JSON+="]"
fi

DATALAKE_ARGS=""
if [[ -n "$RUN_ID" ]]; then
    DATALAKE_ARGS+="run_id=$RUN_ID"
fi
if [[ -n "$STEP" ]]; then
    DATALAKE_ARGS+=",step=$STEP"
fi

# Set HF_UPLOAD_ARG only if UPLOAD_TO_HF is specified
if [[ -n "$UPLOAD_TO_HF" ]]; then
    HF_UPLOAD_ARG="--hf-save-dir ${UPLOAD_TO_HF}//results/${MODEL_NAME_SAFE}"
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
#    "alpaca_eval_v2::tulu" # Removed for high cost of judge
TULU_3_DEV=(
    "gsm8k::tulu"
    "bbh:cot-v1::tulu"
    "drop::llama3"
    "minerva_math::tulu"
    "codex_humaneval::tulu"
    "codex_humanevalplus::tulu"
    "ifeval::tulu"
    "popqa::tulu"
    "mmlu:mc::tulu"
    "mmlu:cot::summarize"
    "alpaca_eval_v3::tulu" # GPT 4.1 judge on OpenAI
    # "alpaca_eval_v4::tulu" # GPT 4.1, judge on Azure 
    "truthfulqa::tulu"
)
TULU_3_UNSEEN=(
    "agi_eval_english:0shot_cot::tulu3"
    "gpqa:0shot_cot::tulu3"
    "mmlu_pro:0shot_cot::tulu3"
    "deepmind_math:0shot_cot-v3::tulu3"
    "bigcodebench_hard::tulu"
    "gpqa:0shot_cot::llama3.1"
    "mmlu_pro:cot::llama3.1"
    "bigcodebench::tulu"
)

# New default task suites
NEXT_MODEL_DEV=(
    # Knowledge
    "mmlu:cot::hamish_zs_reasoning"
    "popqa::hamish_zs_reasoning"
    # "simpleqa::tulu-thinker"
    
    # Reasoning
    "bbh:cot::hamish_zs_reasoning"
    "gpqa:0shot_cot::hamish_zs_reasoning"
    "zebralogic::hamish_zs_reasoning"
    "agi_eval_english:0shot_cot::hamish_zs_reasoning"
    
    # Math
    # [faster] minerva_math_500::hamish_zs_reasoning
    "minerva_math::hamish_zs_reasoning"
    "gsm8k::zs_cot_latex"
    "omega:0-shot-chat"
    "aime::hamish_zs_reasoning"
    # [maybe unseen] aime::hamish_zs_reasoning_2025
    
    # Coding
    "codex_humanevalplus:0-shot-chat::tulu-thinker"
    "mbppplus:0-shot-chat::tulu-thinker"
    "livecodebench_codegeneration::tulu-thinker"
    # [TODO not merged] codeeditorbench
    # [TODO, maybe] cruxeval
    
    # Chat / IF / Vibes
    # "alpaca_eval_v3::hamish_zs_reasoning"
    "ifeval::hamish_zs_reasoning"
    # [expensive, multi-turn all versions] multiturn_alpacaeval::tulu
    # [expensive, typos vibes] styled_evals::tulu
    # [optional, typos compare] styled_math500::tulu
    # [optional, typos compare] styled_popqa::tulu
    # [optional, typos compare] styled_truthfulqa::tulu
)

NEXT_MODEL_UNSEEN=(
    "mmlu_pro:0shot_cot::tulu3"
    # [TODO, not implemented] Humanity's Last Exam
    # [TODO, not implemented] SuperGPQA
    # [TODO, not implemented] BigBenchExtraHard
    "livecodebench_codegeneration::tulu-thinker-hidden"
    "ifbench::tulu"
)

# If custom tasks provided, convert comma-separated string to array
if [[ -n "$CUSTOM_TASKS" ]]; then
    IFS=',' read -ra TASKS <<< "$CUSTOM_TASKS"
else
    # Use the specified task suite or default
    case "$TASK_SUITE" in
        NEXT_MODEL_DEV)
            TASKS=("${NEXT_MODEL_DEV[@]}")
            ;;
        NEXT_MODEL_UNSEEN)
            TASKS=("${NEXT_MODEL_UNSEEN[@]}")
            ;;
        TULU_3_DEV)
            TASKS=("${TULU_3_DEV[@]}")
            ;;
        TULU_3_UNSEEN)
            TASKS=("${TULU_3_UNSEEN[@]}")
            ;;
        *)
            echo "Error: Unknown task suite '$TASK_SUITE'"
            usage
            ;;
    esac
fi

MODEL_TYPE="--model-type vllm"
BATCH_SIZE_VLLM=10000
BATCH_SIZE_OTHER=1
# Set GPU_COUNT and GPU_COUNT_OTHER based on NUM_GPUS
GPU_COUNT="$NUM_GPUS"
GPU_COUNT_OTHER=$((NUM_GPUS * 2))
MODEL_TYPE_OTHER=""

# Build model args JSON with optional process_output
MODEL_ARGS="{\"model_path\":\"${MODEL_LOCATION}\", \"max_length\": ${MAX_LENGTH}, \"trust_remote_code\": \"true\""
if [[ -n "$PROCESS_OUTPUT" ]]; then
    MODEL_ARGS+=", \"process_output\": \"${PROCESS_OUTPUT}\""
fi
MODEL_ARGS+="}"

for TASK in "${TASKS[@]}"; do
    # mmlu and truthfulqa need different batch sizes and gpu counts because they are multiple choice and we cannot use vllm.
    if [[ "$TASK" == "mmlu:mc::tulu" || "$TASK" == "truthfulqa::tulu" ]]; then
        BATCH_SIZE=$BATCH_SIZE_OTHER
        GPU_COUNT=$GPU_COUNT_OTHER
        MODEL_TYPE=$MODEL_TYPE_OTHER
    else
        BATCH_SIZE=$BATCH_SIZE_VLLM
        MODEL_TYPE="--model-type vllm"
        GPU_COUNT="$NUM_GPUS"
    fi

    # NOTE: For gantry args here and below, random numbers like #42 are added to the env variables because they need to be unique names. The numbers are ignored.
    if [ "$EVALUATE_ON_WEKA" == "true" ]; then
        python oe-eval-internal/oe_eval/launch.py \
            --model "$MODEL_NAME" \
            --beaker-workspace "ai2/tulu-3-results" \
            --beaker-budget ai2/oe-adapt \
            --beaker-timeout 48h \
            --task "$TASK" \
            $MODEL_TYPE \
            --batch-size "$BATCH_SIZE" \
            --model-args "$MODEL_ARGS" \
            --task-args "{ \"generation_kwargs\": { \"max_gen_toks\": ${MAX_LENGTH}, \"truncate_context\": false${STOP_SEQUENCES_JSON} } }" \
            ${HF_UPLOAD_ARG} \
            --gpus "$GPU_COUNT" \
            --gantry-args '{"env-secret": "OPENAI_API_KEY=openai_api_key", "weka": "oe-adapt-default:/weka/oe-adapt-default", "env#132":"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1", "env-secret#42": "AZURE_EVAL_API_KEY=azure_eval_api_key"}' \
            ${REVISION_ARG} \
            --cluster "$CLUSTER" \
            --beaker-retries 2 \
            --beaker-image "$BEAKER_IMAGE" \
            --beaker-priority "$PRIORITY" \
            --push-datalake \
            --datalake-tags "$DATALAKE_ARGS"
    else
        python oe-eval-internal/oe_eval/launch.py \
        --model "$MODEL_NAME" \
        --beaker-workspace "ai2/tulu-3-results" \
        --beaker-budget ai2/oe-adapt \
        --beaker-timeout 48h \
        --task "$TASK" \
        $MODEL_TYPE \
        --batch-size "$BATCH_SIZE" \
        --model-args "$MODEL_ARGS" \
        --task-args "{ \"generation_kwargs\": { \"max_gen_toks\": ${MAX_LENGTH}, \"truncate_context\": false${STOP_SEQUENCES_JSON} } }" \
        ${HF_UPLOAD_ARG} \
        --gpus "$GPU_COUNT" \
        --gantry-args "{\"env-secret\": \"OPENAI_API_KEY=openai_api_key\", \"env-secret#43\": \"AZURE_EVAL_API_KEY=azure_eval_api_key\", \"env\":\"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\", \"env-secret#2\":\"HF_TOKEN=HF_TOKEN\", \"mount\": \"/mnt/filestore_1:/filestore\", \"env#111\": \"HF_HOME=/filestore/.cache/huggingface\", \"env#112\": \"HF_DATASETS_CACHE=/filestore/.cache/huggingface\", \"env#113\": \"HF_HUB_CACHE=/filestore/.cache/hub\"}" \
        ${REVISION_ARG} \
        --cluster ai2/augusta-google-1 \
        --beaker-retries 2 \
        --beaker-image "$BEAKER_IMAGE" \
        --beaker-priority  "$PRIORITY" \
        --push-datalake \
        --datalake-tags "$DATALAKE_ARGS"
    fi
done
