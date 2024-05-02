#!/bin/bash

# to do:
model_list=(
    "llama_2_7b-tulu_v2_mix-coding_none"
    "llama_2_7b-tulu_all-coding_none"
    "llama_2_7b-tulu_all-coding_100"
    # "llama_2_7b-tulu_none-coding_100" # in progress
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