#!/bin/bash

# to do:
model_list=(
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.9-llama_2_7b-tulu_none-safety_100-4k_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.8-llama_2_7b-tulu_none-safety_100-4k_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.7-llama_2_7b-tulu_none-safety_100-4k_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.6-llama_2_7b-tulu_none-safety_100-4k_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.5-llama_2_7b-tulu_none-safety_100-4k_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.4-llama_2_7b-tulu_none-safety_100-4k_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.3-llama_2_7b-tulu_none-safety_100-4k_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.2-llama_2_7b-tulu_none-safety_100-4k_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.1-llama_2_7b-tulu_none-safety_100-4k_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-safety_100-4k_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-science_2500-4k_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.9-llama_2_7b-tulu_none-science_2500-4k_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.8-llama_2_7b-tulu_none-science_2500-4k_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.7-llama_2_7b-tulu_none-science_2500-4k_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.6-llama_2_7b-tulu_none-science_2500-4k_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.5-llama_2_7b-tulu_none-science_2500-4k_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.4-llama_2_7b-tulu_none-science_2500-4k_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.3-llama_2_7b-tulu_none-science_2500-4k_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.2-llama_2_7b-tulu_none-science_2500-4k_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.1-llama_2_7b-tulu_none-science_2500-4k_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.9-llama_2_7b-tulu_none-coding_100-4k_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.8-llama_2_7b-tulu_none-coding_100-4k_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.7-llama_2_7b-tulu_none-coding_100-4k_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.6-llama_2_7b-tulu_none-coding_100-4k_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.5-llama_2_7b-tulu_none-coding_100-4k_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.4-llama_2_7b-tulu_none-coding_100-4k_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.3-llama_2_7b-tulu_none-coding_100-4k_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.2-llama_2_7b-tulu_none-coding_100-4k_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_no_science_no_safety_no_coding_0.1-llama_2_7b-tulu_none-coding_100-4k_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_no_science_no_safety_no_coding_1.0-llama_2_7b-tulu_none-coding_100-4k_0.9"
    "llama_2_7b-tulu_none-coding_100-4k"
    "llama_2_7b-tulu_all_no_science_no_safety_no_coding"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular_adaptation \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/tulu_evals/${MODEL}/
done