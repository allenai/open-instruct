#!/bin/bash

# to do:
model_list=(
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.9"
    # "llama_2_7b-tulu_none-coding_100"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b-tulu_none-coding_100_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b-tulu_none-coding_100_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b-tulu_none-coding_100_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b-tulu_none-coding_100_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b-tulu_none-coding_100_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b-tulu_none-coding_100_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b-tulu_none-coding_100_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b-tulu_none-coding_100_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b-tulu_none-coding_100_0.9"
    # "llama_2_7b-tulu_all-coding_100"
    # "tulu_2_7b-tulu_none-coding_100"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b-tulu_none-coding_40_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b-tulu_none-coding_40_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b-tulu_none-coding_40_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b-tulu_none-coding_40_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b-tulu_none-coding_40_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b-tulu_none-coding_40_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b-tulu_none-coding_40_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b-tulu_none-coding_40_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b-tulu_none-coding_40_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b-tulu_none-coding_20_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b-tulu_none-coding_20_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b-tulu_none-coding_20_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b-tulu_none-coding_20_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b-tulu_none-coding_20_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b-tulu_none-coding_20_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b-tulu_none-coding_20_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b-tulu_none-coding_20_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b-tulu_none-coding_20_0.9"
    "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b-tulu_none-coding_80_0.1"
    "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b-tulu_none-coding_80_0.2"
    "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b-tulu_none-coding_80_0.3"
    "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b-tulu_none-coding_80_0.4"
    "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b-tulu_none-coding_80_0.5"
    "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b-tulu_none-coding_80_0.6"
    "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b-tulu_none-coding_80_0.7"
    "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b-tulu_none-coding_80_0.8"
    "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b-tulu_none-coding_80_0.9"

    "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b-tulu_none-coding_60_0.1"
    "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b-tulu_none-coding_60_0.2"
    "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b-tulu_none-coding_60_0.3"
    "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b-tulu_none-coding_60_0.4"
    "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b-tulu_none-coding_60_0.5"
    "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b-tulu_none-coding_60_0.7"
    "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b-tulu_none-coding_60_0.6"
    "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b-tulu_none-coding_60_0.8"
    "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b-tulu_none-coding_60_0.9"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular_adaptation \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned --beaker_image hamishivi/open-instruct-mbpp-test \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/tulu_evals/${MODEL}/
done

# for MODEL in "${model_list[@]}"
# do
# python scripts/submit_eval_jobs.py \
#     --workspace modular-adaptation-coding \
#     --model_name ${MODEL} \
#     --location jacobm \
#     --priority normal \
#     --is_tuned \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/coding/tulu_evals/${MODEL}/
# done