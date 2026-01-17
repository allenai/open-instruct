#!/bin/bash
# ["super_ni", "cot", "flan_v2", "self_instruct", "unnatural_instructions", "stanford_alpaca", "dolly", "sharegpt", "code_alpaca", "gpt4_alpaca", "baize", "oasst1"]
# Run with scripts/data/sft_v1_v2/get_statistics_tulu_v2.sh
# for every dataset, get the statistics
for dataset in super_ni cot flan_v2 self_instruct unnatural_instructions stanford_alpaca dolly sharegpt code_alpaca gpt4_alpaca baize oasst1 lima wizardlm open_orca; do
    echo "Getting statistics for $dataset..."
    python scripts/data/get_statistics.py --data_path data/processed/${dataset}/${dataset}_data.jsonl --save_path data/processed/${dataset}/${dataset}_statistics.json
done
