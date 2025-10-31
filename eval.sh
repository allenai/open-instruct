image_name=michaeln/oe_eval_olmo2_retrofit

model_path_prefix="/weka/oe-adapt-default/allennlp/deletable_checkpoint/saurabhs/grpo_math_from_zero__1__1760546257_checkpoints"

model_name_prefix="grpo_math_from_zero__1__1760546257"

steps=(250 500 750 1000 1250)

tasks="aime:zs_cot_r1::pass_at_32_2025_dapo"

for i in "${!steps[@]}"; do
    REASONER_LENGTH=32768
    step=${steps[$i]}
    model_name=$model_name_prefix"_step_"$step
    model_path=$model_path_prefix/step_$step
    python scripts/submit_eval_jobs.py \
        --model_name $model_name \
        --location $model_path \
        --cluster ai2/jupiter ai2/ceres ai2/saturn ai2/neptune \
        --is_tuned \
        --priority urgent \
        --preemptible \
        --step $step \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --oe_eval_max_length $REASONER_LENGTH \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $tasks \
        --workspace ai2/olmo-instruct \
        --gpu_multiplier 2 \
        --beaker_image $image_name
done


