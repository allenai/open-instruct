#!/bin/bash

# model_name="olmo3-7b_rlzero_olmo3_7b_base__1__1763659491"
# wandb_id="n3irz0v8_16k"
# model_location="gs://ai2-llm/post-training//michaeln//output/olmo3-7b-rlzero-math/checkpoints/${model_name}_checkpoints"
#
# model_name="qwen3_1.7b_rlzero_math__1__1766123249"
model_name="Qwen3-0.6B"
model_location="Qwen/Qwen3-0.6B"
wandb_id="qwen3_0.6b"
# model_location="/weka/oe-adapt-default/allennlp/.cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
# model_name="olmo3_7b_rlzero_math__1__1763966683"
# wandb_id="kn6kewty"
# model_location="gs://ai2-llm/post-training//michaeln/output/${model_name}_checkpoints"
# model_location="/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/${model_name}_checkpoints"
# --model-name "${model_name}_step_${step}" \
# --model-location "${model_location}/step_$step" \

# for step in {100..2000..100}; do
# --step $step \
uv run bash scripts/eval/oe-eval.sh \
    --model-name $model_name \
    --model-location $model_location \
    --beaker-workspace ai2/oe-adapt-code \
    --tasks gsm8k::zs_cot_latex:n32 \
    --run-id https://wandb.ai/ai2-llm/open_instruct_internal/runs/$wandb_id \
    --wandb-run-path ai2-llm/open_instruct_internal/$wandb_id \
    --num_gpus 1 \
    --task-suite NEXT_MODEL_DEV \
    --priority high \
    --process-output r1_style \
    --beaker-image michaeln/oe_eval_internal \
    --cluster 'ai2/neptune'
# done
# --max-length 32768 \
