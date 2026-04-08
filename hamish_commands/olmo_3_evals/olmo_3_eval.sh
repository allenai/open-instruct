
MODEL_PATH=/weka/oe-adapt-default/hamishi/olmo_3_emergency_ckpts/olmo3_dpo_rl_final_mix_jupiter_10108_6245__1__1759426652_checkpoints_step_1000
MODEL_NAME=grabbing_rl_dpo_data_step1000_table20_scores
for run in 1; do
uv run python scripts/submit_eval_jobs.py \
    --model_name $MODEL_NAME \
    --location $MODEL_PATH \
    --cluster ai2/jupiter ai2/ceres \
    --is_tuned \
    --priority high \
    --preemptible \
    --gpu_multiplier 4 \
    --use_hf_tokenizer_template \
    --run_oe_eval_experiments \
    --oe_eval_task_suite "SAFETY_EVAL" \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals
done