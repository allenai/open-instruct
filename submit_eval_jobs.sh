MODEL_PATHS=(
/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo-sft/qwen3-4b-sft-100k/step1476-hf
)
current_evals="alpaca_eval_v3::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning,ifbench::tulu,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite"

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

    # MODEL_NAME=qwen3-1.7b-sft-100k
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals \
        --tokenizer_path /weka/oe-adapt-default/jacobm/repos/cse-579/tokenizers/qwen3-olmo-thinker-eos-old-transformers \
        --oe_eval_tasks $current_evals
done