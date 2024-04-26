#!/bin/bash

# to do:
model_list=(
    # "llama_2_7b-tulu_all-safety_v0_all"

    # "llama_2_7b-tulu_none-safety_20"
    # "llama_2_7b-tulu_none-safety_40"
    # "llama_2_7b-tulu_none-safety_60"
    # "llama_2_7b-tulu_none-safety_80"
    # "llama_2_7b-tulu_none-safety_100"
    # "llama_2_7b-tulu_none-safety_upsample"

    # "llama_2_7b-tulu_all-safety_none"
    # "llama_2_7b-tulu_all-safety_none_seed_123"
    # "llama_2_7b-tulu_all-safety_none_seed_52830"
    # "llama_2_7b-tulu_all-safety_20"
    # "llama_2_7b-tulu_all-safety_40"
    # "llama_2_7b-tulu_all-safety_60"
    # "llama_2_7b-tulu_all-safety_80"
    # "llama_2_7b-tulu_all-safety_100"
    # "llama_2_7b-tulu_all-safety_upsample"

    # "tulu_2_7b_uncensored-tulu_none-safety_20"
    # "tulu_2_7b_uncensored-tulu_none-safety_40"
    # "tulu_2_7b_uncensored-tulu_none-safety_60"
    # "tulu_2_7b_uncensored-tulu_none-safety_80"
    # "tulu_2_7b_uncensored-tulu_none-safety_100"
    # "tulu_2_7b_uncensored-tulu_none-safety_upsample"
    
    # "tulu_2_7b_uncensored-tulu_match-safety_20"
    # "tulu_2_7b_uncensored-tulu_match-safety_40"
    # "tulu_2_7b_uncensored-tulu_match-safety_60"
    # "tulu_2_7b_uncensored-tulu_match-safety_80"
    # "tulu_2_7b_uncensored-tulu_match-safety_100"
    # "tulu_2_7b_uncensored-tulu_match-safety_v0_100"

    # "linear_weighted-llama_2_7b-tulu_all_0.1-safety_20_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-safety_20_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-safety_20_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-safety_20_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-safety_20_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-safety_20_0.4"

    # "linear_weighted-llama_2_7b-tulu_all_0.7-safety_20_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-safety_20_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-safety_20_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-safety_60_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-safety_60_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-safety_60_0.7"
    
    # "linear_weighted-llama_2_7b-tulu_all_0.4-safety_60_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-safety_60_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-safety_60_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-safety_60_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-safety_60_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-safety_60_0.1"
    
    # "linear_weighted-llama_2_7b-tulu_all_0.1-safety_100_0.9"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-safety_100_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-safety_100_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-safety_100_0.6"
    
    # "linear_weighted-llama_2_7b-tulu_all_0.5-safety_100_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-safety_100_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-safety_100_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-safety_100_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-safety_100_0.1"
	
    # "llama_2_7b-tulu_upsample-safety_none"
    # "llama_2_7b-tulu_none-safety_v0_100"
    # "llama_2_7b-tulu_none-safety_10"
    # "linear_weighted-llama_2_7b-tulu_all_0.9-safety_v0_100_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-safety_v0_100_0.2"		
    # "linear_weighted-llama_2_7b-tulu_all_0.7-safety_v0_100_0.3"		
    # "linear_weighted-llama_2_7b-tulu_all_0.6-safety_v0_100_0.4"			
    # "linear_weighted-llama_2_7b-tulu_all_0.5-safety_v0_100_0.5"		
    # "linear_weighted-llama_2_7b-tulu_all_0.4-safety_v0_100_0.6"		
    # "linear_weighted-llama_2_7b-tulu_all_0.3-safety_v0_100_0.7"		
    # "linear_weighted-llama_2_7b-tulu_all_0.2-safety_v0_100_0.8"		
    # "linear_weighted-llama_2_7b-tulu_all_0.1-safety_v0_100_0.9"	

    # "linear_weighted-llama_2_7b-tulu_all_0.9-safety_10_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-safety_10_0.2"		
    # "linear_weighted-llama_2_7b-tulu_all_0.7-safety_10_0.3"		
    # "linear_weighted-llama_2_7b-tulu_all_0.6-safety_10_0.4"		
    # "linear_weighted-llama_2_7b-tulu_all_0.5-safety_10_0.5"		
    # "linear_weighted-llama_2_7b-tulu_all_0.4-safety_10_0.6"		
    # "linear_weighted-llama_2_7b-tulu_all_0.3-safety_10_0.7"		
    # "linear_weighted-llama_2_7b-tulu_all_0.2-safety_10_0.8"		
    # "linear_weighted-llama_2_7b-tulu_all_0.1-safety_10_0.9"	

    # "linear_weighted-llama_2_7b-tulu_all_0.9-tulu_2_7b_uncensored_safety_100_0.1"
    # "linear_weighted-llama_2_7b-tulu_all_0.8-tulu_2_7b_uncensored_safety_100_0.2"
    # "linear_weighted-llama_2_7b-tulu_all_0.7-tulu_2_7b_uncensored_safety_100_0.3"
    # "linear_weighted-llama_2_7b-tulu_all_0.6-tulu_2_7b_uncensored_safety_100_0.4"
    # "linear_weighted-llama_2_7b-tulu_all_0.5-tulu_2_7b_uncensored_safety_100_0.5"
    # "linear_weighted-llama_2_7b-tulu_all_0.4-tulu_2_7b_uncensored_safety_100_0.6"
    # "linear_weighted-llama_2_7b-tulu_all_0.3-tulu_2_7b_uncensored_safety_100_0.7"
    # "linear_weighted-llama_2_7b-tulu_all_0.2-tulu_2_7b_uncensored_safety_100_0.8"
    # "linear_weighted-llama_2_7b-tulu_all_0.1-tulu_2_7b_uncensored_safety_100_0.9"

    # "llama_2_7b-tulu_all-safety_10"
    # "tulu_2_7b_uncensored-tulu_match-safety_10"
    # "tulu_2_7b_uncensored-tulu_none-safety_10"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_1.0"

    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.1"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.2"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.3"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.4"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.5"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.6"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.7"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.8"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.9"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_1.0"

    # "linear_weighted-llama_2_7b-tulu_all_0.98-safety_10_0.02"
    # "linear_weighted-llama_2_7b-tulu_all_0.96-safety_20_0.04"
    # "linear_weighted-llama_2_7b-tulu_all_0.88-safety_60_0.12"
    # "linear_weighted-llama_2_7b-tulu_all_0.82-safety_100_0.18"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_10_0.02"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_20_0.04"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_60_0.13"
    # "task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.22"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.1"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.2"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.22"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.3"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.4"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.5"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.6"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.7"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.8"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_0.9"
    "dare_task_arithmetic-llama_2_7b-tulu_all_1.0-safety_100_1.0"
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