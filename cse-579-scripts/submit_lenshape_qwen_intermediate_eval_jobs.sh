# Eval submitter for our Qwen length-shaping run at intermediate steps (100..900).
# step_1000 was already evaluated by submit_lenshape_qwen_eval_jobs.sh — not
# re-submitted here. These run at normal priority to avoid clogging urgent
# slots; lower-priority is fine for plotting training curves later.

MODEL_PATHS=(
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_100"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_200"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_300"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_400"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_500"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_600"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_700"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_800"
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/step_900"
)
current_evals="alpaca_eval_v3::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning,ifbench::tulu,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2025_deepseek"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")

    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        STEP_NUM="$BASENAME"
        EXPERIMENT_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR" _checkpoints)
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi

    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/olmo-instruct \
        --priority normal \
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
