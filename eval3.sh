#image_name=oe-eval-beaker/oe_eval_olmo2_retrofit_auto
image_name=michaeln/oe_eval_olmo2_retrofit
gcloud_base="gs://ai2-llm/post-training//faezeb/output/grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_checkpoints"

steps=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950)

tasks="livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"

for step in "${steps[@]}"; do
    MODEL_PATH="${gcloud_base}/step_${step}"
    MODEL_NAME="grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_step_${step}"
    REASONER_LENGTH=32768
    python scripts/submit_eval_jobs.py \
        --model_name $MODEL_NAME \
        --location $MODEL_PATH \
        --cluster ai2/jupiter ai2/ceres ai2/saturn ai2/neptune \
        --is_tuned \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --oe_eval_max_length $REASONER_LENGTH \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $tasks \
        --workspace ai2/tulu-3-results \
        --gpu_multiplier 2 \
        --beaker_image $image_name
done


