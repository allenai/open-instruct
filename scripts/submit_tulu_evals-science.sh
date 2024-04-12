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
    "llama_2_7b-tulu_76007-science_304028"
)

# for tuple in "${model_list[@]}"
for MODEL in "${model_list[@]}"
do

python scripts/submit_eval_jobs.py \
    --workspace modular-adaptation-science \
    --model_name ${MODEL} \
    --location jacobm \
    --priority low \
    --is_tuned \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu_evals/${MODEL}/
done