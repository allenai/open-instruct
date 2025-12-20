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
    echo "TASK_SUITE should be one of: NEXT_MODEL_DEV, NEXT_MODEL_UNSEEN, TULU_3_DEV, TULU_3_UNSEEN, SAFETY_EVAL, SAFETY_EVAL_REASONING (default: NEXT_MODEL_DEV)"
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
        --wandb-run-path) WANDB_RUN_PATH="$2"; shift ;;
        --stop-sequences) STOP_SEQUENCES="$2"; shift ;;
        --beaker-image) BEAKER_IMAGE="$2"; shift ;;
        --cluster) CLUSTER="$2"; shift ;;
        --process-output) PROCESS_OUTPUT="$2"; shift ;;
        --beaker-workspace) BEAKER_WORKSPACE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Default beaker workspace if not provided; does not override user input.
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/tulu-3-results}"
if ! [[ "$BEAKER_WORKSPACE" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Error: --beaker-workspace must be fully qualified as '<org>/<workspace>' (e.g., 'ai2/tulu-3-results'). Received: '$BEAKER_WORKSPACE'"
    exit 1
fi

# cluster/weka mount logic: default true (to use non-augusta)
# if model starts with gs://, set evaluate_on_weka to false.
# All the logic is now handled internally, the flag is useless but keeping for backwards compatibility since people have scripts with it
EVALUATE_ON_WEKA="true"
if [[ "$MODEL_LOCATION" == gs://* ]]; then
    EVALUATE_ON_WEKA="false"
fi

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
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
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

# Set wandb run path to upload to wandb if available
WANDB_ARG=""
if [[ -n "$WANDB_RUN_PATH" ]]; then
    beaker_user=$(beaker account whoami --format text | awk 'NR==2 {print $2}')
    echo "Using WANDB_API_KEY from ${beaker_user}"
    if ! beaker secret list --workspace "$BEAKER_WORKSPACE" | grep -q "${beaker_user}_WANDB_API_KEY"; then
        echo "WARNING: No ${beaker_user}_WANDB_API_KEY secret found in workspace $BEAKER_WORKSPACE."
        echo "add your WANDB_API_KEY as a secret to this workspace in order to log oe-eval results to wandb"
    else
        WANDB_ARG=" --wandb-run-path $WANDB_RUN_PATH --gantry-secret-wandb-api-key ${beaker_user}_WANDB_API_KEY"
    fi
fi

DATALAKE_ARGS=""
if [[ -n "$RUN_ID" ]]; then
    DATALAKE_ARGS+="run_id=$RUN_ID"
fi
if [[ -n "$STEP" ]]; then
    DATALAKE_ARGS+=",step=$STEP"
    if [[ -n "$WANDB_ARG" ]]; then
        WANDB_ARG+=" --wandb-run-step $STEP"
    fi
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
    "mmlu:cot::hamish_zs_reasoning_deepseek"
    "popqa::hamish_zs_reasoning_deepseek"
    "simpleqa::tulu-thinker_deepseek"

    # Reasoning
    "bbh:cot::hamish_zs_reasoning_deepseek_v2" # OLD: "bbh:cot::hamish_zs_reasoning_deepseek"
    "gpqa:0shot_cot::qwen3-instruct"
    "zebralogic::hamish_zs_reasoning_deepseek"
    "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"

    # Math
    # [faster] minerva_math_500::hamish_zs_reasoning
    "minerva_math::hamish_zs_reasoning_deepseek"
    "gsm8k::zs_cot_latex_deepseek"
    "omega_500:0-shot-chat_deepseek" # OLD: "omega:0-shot-chat"
    "aime:zs_cot_r1::pass_at_32_2024_deepseek"
    "aime:zs_cot_r1::pass_at_32_2025_deepseek"  # OLD: "aime::hamish_zs_reasoning"

    # Coding
    "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek"
    "mbppplus:0-shot-chat::tulu-thinker_deepseek"
    "livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
    # [TODO not merged] codeeditorbench - requires separate server
    # [TODO, maybe] cruxeval

    # Chat / IF / Vibes
    "alpaca_eval_v3::hamish_zs_reasoning_deepseek"
    "ifeval::hamish_zs_reasoning_deepseek"
    # [expensive, multi-turn all versions] multiturn_alpacaeval::tulu
    # [expensive, typos vibes] styled_evals::tulu
    # [optional, typos compare] styled_math500::tulu
    # [optional, typos compare] styled_popqa::tulu
    # [optional, typos compare] styled_truthfulqa::tulu

    # Tool Use
    "bfcl_all::std" # This requires special logic on model_args and metadata, handled below
)

NEXT_MODEL_UNSEEN=(
    "mmlu_pro:0shot_cot::tulu3"
    # [TODO, not implemented] Humanity's Last Exam
    # [TODO, not implemented] SuperGPQA
    # [TODO, not implemented] BigBenchExtraHard
    "livecodebench_codegeneration::tulu-thinker-hidden_no_think_tags"
    "ifbench::tulu"
)

SAFETY_EVAL=(
    "safety::olmo3"
)

SAFETY_EVAL_REASONING=(
    "safety_reasoning::olmo3"
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
        SAFETY_EVAL)
            TASKS=("${SAFETY_EVAL[@]}")
            ;;
        SAFETY_EVAL_REASONING)
            TASKS=("${SAFETY_EVAL_REASONING[@]}")
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
# Source the non-AI2 models list for bfcl
source "$(dirname "$0")/bfcl_supported_models.sh"

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

    # Handle special case for bfcl_all::std task - requires additional metadata
    if [[ "$TASK" == "bfcl_all::std" ]]; then
        # Update MODEL_ARGS
        # Check if MODEL_LOCATION is a local model (starts with beaker:// or /weka/ or gs://)
            if [[ "$MODEL_LOCATION" == beaker://* ]] || [[ "$MODEL_LOCATION" == /weka/* ]] || [[ "$MODEL_LOCATION" == gs://* ]]; then
            # Local model: keep model_path as MODEL_LOCATION, add metadata with allenai/general-tool-use-dev
            MODEL_ARGS="${MODEL_ARGS%?}, \"metadata\": {\"extra_eval_config\": {\"model_name\": \"allenai/general-tool-use-dev\"}}}"
        else
            # HF model: check if it's supported
            if [[ " ${SUPPORTED_MODELS[*]} " =~ " ${MODEL_LOCATION} " ]]; then
                # Supported HF model: remove model_path, no metadata needed
                BASE_ARGS="{\"max_length\": ${MAX_LENGTH}, \"trust_remote_code\": \"true\""
                if [[ -n "$PROCESS_OUTPUT" ]]; then
                    BASE_ARGS+=", \"process_output\": \"${PROCESS_OUTPUT}\""
                fi
                BASE_ARGS+="}"
                MODEL_ARGS="$BASE_ARGS"

                # remove hf- from model name
                MODEL_NAME="${MODEL_NAME#hf-}"
            else
                # Unsupported HF model: skip this task
                echo "Warning: Model '${MODEL_LOCATION}' is not supported for bfcl_all::std task. Skipping..."
                continue
            fi
        fi
        # Update env length variable in gantry-args, needed for bfcl
        MAX_TOKENS_ARG=", \"env#111\": \"MAX_TOKENS=${MAX_LENGTH}\""
    else
        # For other tasks, use the original MODEL_ARGS without metadata, and no gantry-arg for length
        MAX_TOKENS_ARG=""
    fi

    # NOTE: For gantry args here and below, random numbers like #42 are added to the env variables because they need to be unique names. The numbers are ignored.
    # Build gantry args
    if [ "$EVALUATE_ON_WEKA" == "true" ]; then
        GANTRY_ARGS="{\"env-secret\": \"OPENAI_API_KEY=openai_api_key\", \"weka\": \"oe-adapt-default:/weka/oe-adapt-default\", \"weka#44\": \"oe-training-default:/weka/oe-training-default\", \"env-secret#2\":\"HF_TOKEN=HF_TOKEN\", \"env#132\":\"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\", \"env-secret#42\": \"AZURE_EVAL_API_KEY=azure_eval_api_key\"${MAX_TOKENS_ARG}}"
    else
        GANTRY_ARGS="{\"env-secret\": \"OPENAI_API_KEY=openai_api_key\", \"env-secret#43\": \"AZURE_EVAL_API_KEY=azure_eval_api_key\", \"env\":\"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\", \"env-secret#2\":\"HF_TOKEN=HF_TOKEN\", \"mount\": \"/mnt/filestore_1:/filestore\", \"env#111\": \"HF_HOME=/filestore/.cache/huggingface\", \"env#112\": \"HF_DATASETS_CACHE=/filestore/.cache/huggingface\", \"env#113\": \"HF_HUB_CACHE=/filestore/.cache/hub\"${MAX_TOKENS_ARG}}"
    fi

    if [ "$EVALUATE_ON_WEKA" == "true" ]; then
        python oe-eval-internal/oe_eval/launch.py \
            --model "$MODEL_NAME" \
            --beaker-workspace "$BEAKER_WORKSPACE" \
            --beaker-budget ai2/oe-adapt \
            --beaker-timeout 48h \
            --task "$TASK" \
            $MODEL_TYPE \
            --batch-size "$BATCH_SIZE" \
            --model-args "$MODEL_ARGS" \
            --task-args "{ \"generation_kwargs\": { \"max_gen_toks\": ${MAX_LENGTH}, \"truncate_context\": false${STOP_SEQUENCES_JSON} } }" \
            ${HF_UPLOAD_ARG} \
            --gpus "$GPU_COUNT" \
            --gantry-args "$GANTRY_ARGS" \
            ${REVISION_ARG} \
            ${WANDB_ARG} \
            --cluster "$CLUSTER" \
            --beaker-retries 2 \
            --beaker-image "$BEAKER_IMAGE" \
            --beaker-priority "$PRIORITY" \
            --push-datalake \
            --datalake-tags "$DATALAKE_ARGS"
    else
        python oe-eval-internal/oe_eval/launch.py \
        --model "$MODEL_NAME" \
        --beaker-workspace "$BEAKER_WORKSPACE" \
        --beaker-budget ai2/oe-adapt \
        --beaker-timeout 48h \
        --task "$TASK" \
        $MODEL_TYPE \
        --batch-size "$BATCH_SIZE" \
        --model-args "$MODEL_ARGS" \
        --task-args "{ \"generation_kwargs\": { \"max_gen_toks\": ${MAX_LENGTH}, \"truncate_context\": false${STOP_SEQUENCES_JSON} } }" \
        ${HF_UPLOAD_ARG} \
        --gpus "$GPU_COUNT" \
        --gantry-args "$GANTRY_ARGS" \
        ${REVISION_ARG} \
        ${WANDB_ARG} \
        --cluster ai2/augusta \
        --beaker-retries 2 \
        --beaker-image "$BEAKER_IMAGE" \
        --beaker-priority  "$PRIORITY" \
        --push-datalake \
        --datalake-tags "$DATALAKE_ARGS"
    fi
done
