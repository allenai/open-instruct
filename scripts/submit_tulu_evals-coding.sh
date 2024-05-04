#!/bin/bash

# to do:
model_list=(
    # "llama_2_7b-tulu_v2_mix-coding_none"
    # "llama_2_7b-tulu_all-coding_none"
    # "llama_2_7b-tulu_all-coding_50"
    # "llama_2_7b-tulu_all-coding_100"
    # "llama_2_7b-tulu_none-coding_50"
    # "llama_2_7b-tulu_none-coding_100"
    # "tulu_2_code_none-tulu_none-coding_50"
    "tulu_2_13b_retrain"

    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.51"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_100_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.26"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-coding_50_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-coding_100_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-coding_100_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-coding_100_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.66-coding_100_0.34"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-coding_100_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-coding_100_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-coding_100_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-coding_100_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-coding_100_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-coding_100_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-coding_50_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-coding_50_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-coding_50_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-coding_50_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-coding_50_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-coding_50_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-coding_50_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-coding_50_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-coding_50_0.9"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-coding \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned --beaker_image hamishivi/open-instruct-mbpp-test \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/coding/tulu_evals/${MODEL}/
done

for MODEL in "${model_list[@]}"
do
python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-coding \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/coding/tulu_evals/${MODEL}/
done