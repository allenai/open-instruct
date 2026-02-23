#!/bin/bash

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128-hf"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")

    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi

    echo "=========================================="
    echo "Submitting eval for: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "=========================================="

    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}-updated-evals" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 \
        --s3_output_dir s3://ai2-sewonm/sanjaya/post_training_eval_results/${MODEL_NAME}/

    if [ $? -eq 0 ]; then
        echo "Successfully submitted evaluation for $MODEL_NAME"
    else
        echo "Failed to submit evaluation for $MODEL_NAME"
        echo "Continuing with next model..."
    fi

    echo ""
done

echo "=========================================="
echo "All evaluations submitted!"
echo "=========================================="
