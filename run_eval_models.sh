#!/bin/bash

MODEL_PATHS=(
    "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_5e-5/step1534-hf"
    "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_8e-5/step1534-hf"
    "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_1e-4/step1534-hf"
    "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_2e-4/step1534-hf"
    "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_5e-4/step1534-hf"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")

    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi

    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}-updated-evals" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn-cirrascale \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done
