set -ex

MODEL_PATH=$1

for dataset in hotpotqa nq tqa 2wiki; do
    python -m open_instruct.search_utils.short_form_qa_eval \
        --dataset_name $dataset \
        --model_path ${MODEL_PATH} \
        --output_dir ${1}_${dataset}
done
