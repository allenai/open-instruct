#!/bin/bash

# Define base output directory
BASE_OUTPUT_DIR="s3://ai2-sewonm/sanjaya/post_training_eval_results"

# Define model paths and names arrays
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/math_expert/step594-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/code_expert/step150-hf"
)

MODEL_NAMES=(
    "SFTed-Math-Expert"
    "SFTed-Code-Expert"
)

# Check that arrays have the same length
if [ ${#MODEL_PATHS[@]} -ne ${#MODEL_NAMES[@]} ]; then
    echo "Error: MODEL_PATHS and MODEL_NAMES arrays have different lengths"
    exit 1
fi

# Iterate through each model
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo "=========================================="
    echo "Processing model $((i+1))/${#MODEL_PATHS[@]}"
    echo "Model Name: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "=========================================="
    
    # Run the evaluation command
    python scripts/submit_eval_jobs.py \
        --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/ceres-cirrascale \
        --is_tuned \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --workspace tulu-3-results \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --beaker_image jacobm/flex-olmo-oe-eval-vllm \
        --skip_oi_evals \
        --gpu_multiplier 2 \
        --oe_eval_tasks mmlu:cot::hamish_zs_reasoning,popqa::hamish_zs_reasoning,simpleqa::tulu-thinker,bbh:cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,codex_humanevalplus:0-shot-chat::tulu-thinker,mbppplus:0-shot-chat::tulu-thinker,alpaca_eval_v3::hamish_zs_reasoning,ifeval::hamish_zs_reasoning \
        --s3_output_dir "${BASE_OUTPUT_DIR}/${MODEL_NAME}/" \
        --location "$MODEL_PATH" \
        --model_name "$MODEL_NAME-temp_suite"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully submitted evaluation job for $MODEL_NAME"
    else
        echo "❌ Failed to submit evaluation job for $MODEL_NAME"
        echo "Continuing with next model..."
    fi
    
    echo ""
done

echo "=========================================="
echo "All evaluation jobs have been processed!"
echo "=========================================="