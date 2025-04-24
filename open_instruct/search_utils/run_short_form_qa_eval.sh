#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 1 ] || { echo "Usage: $0 MODEL_PATH"; exit 1; }
MODEL_PATH=$1

# define mason command as an array for safety
read -r -a MASON_CMD <<< \
  "python mason.py \
    --env MASSIVE_DS_URL=http://saturn-cs-aus-234.reviz.ai2.in:40155/search \
    --image hamishivi/1904_rl_rag \
    --pure_docker_mode \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --priority high \
    --gpus 1 \
    --workspace ai2/rl-rag \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale \
    --"

# ensure output dirs exist
for dataset in hotpotqa nq tqa 2wiki simpleqa; do
  outdir="${MODEL_PATH}/${dataset}_results"
    "${MASON_CMD[@]}" \python -m open_instruct.search_utils.short_form_qa_eval \
        --dataset_name "$dataset" \
        --model_path "$MODEL_PATH" \
        --num_docs 3 \
        --output_dir "$outdir"
done
