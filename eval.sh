
MODEL_NAMES=(
    "qwen2.5-7b-ot3"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    WANDB_RUN=placeholder
    python scripts/submit_eval_jobs.py \
            --model_name $MODEL_NAME \
            --location allenai/open_instruct_dev \
            --hf_revision qwen_2p5_7b-open_thoughts_3__8__1751916783 \
            --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
            --is_tuned \
            --priority high \
            --preemptible \
            --use_hf_tokenizer_template \
            --run_oe_eval_experiments \
            --evaluate_on_weka \
            --run_id $WANDB_RUN \
            --workspace tulu-3-results \
            --oe_eval_max_length 32768 \
            --oe_eval_stop_sequences '</answer>' \
            --process_output r1_style  \
            --skip_oi_evals 
done