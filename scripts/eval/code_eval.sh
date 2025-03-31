url=https://wandb.ai/ai2-llm/open_instruct_internal/runs/rfriujym
exp_name=0328_tulu3_8b_grpo_fast_code_27519__1__1743128548
for step in 40 80 120 160 200 240 280 320 360 400 440 480 520 560 600 640 680 720 760 800 840 880 920 960 1000 1040 1080 1120 1160 1200 1240 1280 1320 1360 1400 1440 1480 1520 1560 1600 1640 1680 1720 1760 1800 1840 1880 1920 1960 2000 2040 2080 2120 2160 2200 2240 2280 2320 2360 2400 2440 2480 2520 2560 2600; do
python scripts/submit_eval_jobs.py \
    --model_name ${exp_name}_step_${step} \
    --location /weka/oe-adapt-default/allennlp/deletable_checkpoint/saurabhs/${exp_name}_checkpoints/step_$step \
    --cluster ai2/jupiter-cirrascale-2 ai2/neptune-cirrascale ai2/saturn-cirrascale ai2/ceres-cirrascale  \
    --is_tuned \
    --workspace tulu-3-results \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image nathanl/open_instruct_auto \
    --oe_eval_max_length 16384 \
    --oe_eval_tasks gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct \
    --skip_oi_evals \
    --gpu_multiplier 1 \
    --evaluate_on_weka \
    --step $step \
    --run_id $url \
    --run_oe_eval_experiments
done