#!/bin/bash

# to do:
model_list=(
    # "llama_2_7b-tulu_all-science_2500_minus_science_2500"
    # "llama_2_7b-tulu_all-science_2500_minus_tulu"
    # "llama_2_7b-tulu_0-science_380035"
    # "llama_2_7b-tulu_38004-science_342031"
    # "llama_2_7b-tulu_114011-science_266024"
    # "llama_2_7b-tulu_152014-science_228021"
    # "llama_2_7b-tulu_190018-science_190017"
    # "llama_2_7b-tulu_228021-science_152014"
    # "llama_2_7b-tulu_266025-science_114010"
    # "llama_2_7b-tulu_304028-science_76007"
    # "llama_2_7b-tulu_380035-science_0"
    # "llama_2_7b-tulu_76007-science_304028"

    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_1.0"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.1"

    # "task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.19"
    # "linear_weighted-llama_2_7b-tulu_all_0.84-science_2500_0.16"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b_science_2500_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b_science_2500_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b_science_2500_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b_science_2500_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b_science_2500_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b_science_2500_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b_science_2500_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b_science_2500_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b_science_2500_0.9"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.1"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.19"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.2"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.3"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.4"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.5"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.6"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.7"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.8"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_0.9"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-science_2500_1.0"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-science \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu_evals/${MODEL}/
done