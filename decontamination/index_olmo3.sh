usable_datasets=(
    "jacobmorrison/OpenThoughts3-456k-no-cot"
    "saurabh5/correct-python-sft-187k"
    "allenai/IF_sft_data_verified_permissive"
    "jacobmorrison/verifiable-tasks-o3-7500"
    "jacobmorrison/valpy_if_qwq_reasoning_verified_no_reasoning"
    "saumyamalik/Wildchat-1M-gpt-4.1-regenerated-english"
    "jacobmorrison/Wildchat-1m-gpt-4.1-regeneration-not-english"
    "ai2-adapt-dev/toolu-sft-mix-T2"
)
reasoner_datasets=(
    "allenai/OpenThoughts3-merged-cn-fltrd-final-ngram-filtered-chinese-filtered"
    "allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered-chinese-filtered"
    "allenai/wildchat-r1-p2-repetition-filter-chinese-filtered"
)

for dataset in "${usable_datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python index.py --dataset "$dataset"
done
