set -ex

RESULTS=$1

for dataset in hotpotqa nq tqa 2wiki simpleqa; do
    python -m open_instruct.search_utils.naive_rag_baseline_eval \
        --dataset_name $dataset \
        --model_path ai2-adapt-dev/tulu_3_long_finetune_qwen_7b_reg \
        --output_dir ${RESULTS}/${dataset}_naive_rag_results
done
