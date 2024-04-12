#!/bin/bash

# to do:
model_list=(
    # "llama_2_7b-tulu_none-safety_20"
    # "llama_2_7b-tulu_none-safety_40"
    # "llama_2_7b-tulu_none-safety_60"
    # "llama_2_7b-tulu_none-safety_80"
    # "llama_2_7b-tulu_none-safety_100"
    # "llama_2_7b-tulu_none-safety_upsample"
    # "llama_2_7b-tulu_all-safety_none_seed_123"
    # "llama_2_7b-tulu_all-safety_none_seed_52830"
    # "llama_2_7b-tulu_all-safety_80"
    # "llama_2_7b-tulu_all-safety_100"
    # "llama_2_7b-tulu_all-safety_upsample"
    # "tulu_2_7b_uncensored-tulu_none-safety_40"
    # "tulu_2_7b_uncensored-tulu_none-safety_60"
    # "tulu_2_7b_uncensored-tulu_none-safety_80"
    # "tulu_2_7b_uncensored-tulu_none-safety_upsample"
    # "tulu_2_7b_uncensored-tulu_match-safety_100"
    # "tulu_2_7b_uncensored-tulu_match-safety_v0_100"
    "linear_weighted-llama_2_7b-tulu_all_0.1-safety_20_0.9"
    "linear_weighted-llama_2_7b-tulu_all_0.2-safety_20_0.8"
    "linear_weighted-llama_2_7b-tulu_all_0.3-safety_20_0.7"
    "linear_weighted-llama_2_7b-tulu_all_0.4-safety_20_0.6"
    "linear_weighted-llama_2_7b-tulu_all_0.5-safety_20_0.5"
    "linear_weighted-llama_2_7b-tulu_all_0.6-safety_20_0.4"
    "linear_weighted-llama_2_7b-tulu_all_0.7-safety_20_0.3"
    "linear_weighted-llama_2_7b-tulu_all_0.8-safety_20_0.2"
    "linear_weighted-llama_2_7b-tulu_all_0.9-safety_20_0.1"
    "linear_weighted-llama_2_7b-tulu_all_0.1-safety_60_0.9"
    "linear_weighted-llama_2_7b-tulu_all_0.2-safety_60_0.8"
    "linear_weighted-llama_2_7b-tulu_all_0.3-safety_60_0.7"
    "linear_weighted-llama_2_7b-tulu_all_0.4-safety_60_0.6"
    "linear_weighted-llama_2_7b-tulu_all_0.5-safety_60_0.5"
    "linear_weighted-llama_2_7b-tulu_all_0.6-safety_60_0.4"
    "linear_weighted-llama_2_7b-tulu_all_0.7-safety_60_0.3"
    "linear_weighted-llama_2_7b-tulu_all_0.8-safety_60_0.2"
    "linear_weighted-llama_2_7b-tulu_all_0.9-safety_60_0.1"
    "linear_weighted-llama_2_7b-tulu_all_0.1-safety_100_0.9"
    "linear_weighted-llama_2_7b-tulu_all_0.2-safety_100_0.8"
    "linear_weighted-llama_2_7b-tulu_all_0.3-safety_100_0.7"
    "linear_weighted-llama_2_7b-tulu_all_0.4-safety_100_0.6"
    "linear_weighted-llama_2_7b-tulu_all_0.5-safety_100_0.5"
    "linear_weighted-llama_2_7b-tulu_all_0.6-safety_100_0.4"
    "linear_weighted-llama_2_7b-tulu_all_0.7-safety_100_0.3"
    "linear_weighted-llama_2_7b-tulu_all_0.8-safety_100_0.2"
    "linear_weighted-llama_2_7b-tulu_all_0.9-safety_100_0.1"
)

# for tuple in "${model_list[@]}"
# do
# IFS=',' read -r BEAKER_DATASET MODEL <<< "$tuple"

for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-safety \
    --model_name ${MODEL} \
    --location jacobm \
    --priority normal \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu_evals/${MODEL}/
done