#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 1 ] || { echo "Usage: $0 MODEL_PATH"; exit 1; }
MODEL_PATH=$1

# ensure output dirs exist
for dataset in hotpotqa nq tqa 2wiki simpleqa; do
    echo "Showing results for $dataset"
    python -m open_instruct.search_utils.short_form_qa_eval \
        --dataset_name "$dataset" \
        --model_path "$MODEL_PATH" \
        --num_docs 3 \
        --output_dir tmp \
        --analyse_existing ${MODEL_PATH}/${dataset}_results/predictions.jsonl
done
rm -rf tmp
