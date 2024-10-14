#!/bin/bash
# Run with scripts/data/get_statistics_tulu_v3.sh

# List of datasets
datasets=(
    natolambert/tulu-v2-sft-mixture-flan
    natolambert/tulu-v2-sft-mixture-cot
    ai2-adapt-dev/personahub_math_v4_149975
    AI-MO/NuminaMath-TIR
    HuggingFaceH4/no_robots
    allenai/openassistant-guanaco-reformatted
    ai2-adapt-dev/tulu_hard_coded_examples
    ai2-adapt-dev/SciRIFF-train-mix-science
    ai2-adapt-dev/Table-GPT-All-train
    ai2-adapt-dev/personahub_ifdata_v1_29980
    ai2-adapt-dev/coconot-sft-reformat
    ai2-adapt-dev/openmath-2-gsm8k
    m-a-p/CodeFeedback-Filtered-Instruction
    ai2-adapt-dev/WildChat-1M-Full-GPT4-Only
    ai2-adapt-dev/synthetic-finalresp-wildguarmixtrain
    ai2-adapt-dev/processed-wildjailbreak
)

# For every dataset, get the statistics
for dataset in "${datasets[@]}"; do
    echo "Getting statistics for $dataset..."
    python open_instruct/get_statistics.py --data_path ${dataset} --save_path data/processed/${dataset}_statistics.json
done
