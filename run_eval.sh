#!/bin/bash

# model_name="olmo3-7b_rlzero_olmo3_7b_base__1__1763659491"
# wandb_id="n3irz0v8_16k"
# model_location="gs://ai2-llm/post-training//michaeln//output/olmo3-7b-rlzero-math/checkpoints/${model_name}_checkpoints"
#
model_name="qwen3_1.7b_rlzero_math__1__1766123249"
wandb_id="op75161d"
# model_name="olmo3_7b_rlzero_math__1__1763966683"
# wandb_id="kn6kewty"
# model_location="gs://ai2-llm/post-training//michaeln/output/${model_name}_checkpoints"
model_location="/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/${model_name}_checkpoints"


# for step in {100..2000..100}; do
    uv run bash scripts/eval/oe-eval.sh \
        --model-name "${model_name}_step_${step}" \
        --model-location "${model_location}/step_$step" \
        --beaker-workspace ai2/tulu-3-results \
        --tasks aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero,minerva_math_500::hamish_zs_reasoning_rlzero \
        --run-id https://wandb.ai/ai2-llm/open_instruct_internal/runs/$wandb_id \
        --step $step \
        --wandb-run-path ai2-llm/open_instruct_internal/$wandb_id \
        --num_gpus 1 \
        --max-length 32768 \
        --task-suite NEXT_MODEL_DEV \
        --priority normal \
        --process-output r1_style \
        --beaker-image oe-eval-beaker/oe_eval_auto \
        --cluster 'ai2/neptune'
done
