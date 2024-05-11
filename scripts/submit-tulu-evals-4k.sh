#!/bin/bash

# to do:
model_list=(
    # "llama_2_7b-tulu_all_no_science_no_safety_no_coding-science_2500"
    # "llama_2_7b-tulu_all_no_science_no_safety_no_coding-coding_100"
    # "llama_2_7b-tulu_all_no_science_no_safety_no_coding-safety_100"
    # "finetune-tulu_2_7b_no_science_no_safety_no_coding-tulu_match-coding_100-4k"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.9-tulu_2_7b-tulu_none-science_2500_0.1"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.8-tulu_2_7b-tulu_none-science_2500_0.2"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.7-tulu_2_7b-tulu_none-science_2500_0.3"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.6-tulu_2_7b-tulu_none-science_2500_0.4"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.5-tulu_2_7b-tulu_none-science_2500_0.5"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.4-tulu_2_7b-tulu_none-science_2500_0.6"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.3-tulu_2_7b-tulu_none-science_2500_0.7"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.2-tulu_2_7b-tulu_none-science_2500_0.8"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.1-tulu_2_7b-tulu_none-science_2500_0.9"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.9-tulu_2_7b-tulu_none-safety_100_0.1"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.8-tulu_2_7b-tulu_none-safety_100_0.2"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.7-tulu_2_7b-tulu_none-safety_100_0.3"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.6-tulu_2_7b-tulu_none-safety_100_0.4"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.5-tulu_2_7b-tulu_none-safety_100_0.5"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.4-tulu_2_7b-tulu_none-safety_100_0.6"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.3-tulu_2_7b-tulu_none-safety_100_0.7"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.2-tulu_2_7b-tulu_none-safety_100_0.8"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.1-tulu_2_7b-tulu_none-safety_100_0.9"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.9-tulu_2_7b-tulu_none-coding_100_0.1"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.8-tulu_2_7b-tulu_none-coding_100_0.2"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.7-tulu_2_7b-tulu_none-coding_100_0.3"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.6-tulu_2_7b-tulu_none-coding_100_0.4"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.5-tulu_2_7b-tulu_none-coding_100_0.5"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.4-tulu_2_7b-tulu_none-coding_100_0.6"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.3-tulu_2_7b-tulu_none-coding_100_0.7"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.2-tulu_2_7b-tulu_none-coding_100_0.8"
    "linear_weighted-llama_2_7b-tulu_consistent_mix_0.1-tulu_2_7b-tulu_none-coding_100_0.9"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-safety \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/tulu_evals/${MODEL}/
done