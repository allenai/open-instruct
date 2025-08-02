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
    "allenai/wildchat-r1-p2-repetition-filter-chinese-filtered"
    "allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered"
    "allenai/valpy_if_qwq_reasoning_verified-keyword-filter-datecutoff-chinese-ngram-content-filtered"
    "allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered"
    "allenai/the-algorithm-python-r1-format-filtered-keyword-filtered-filter-datecutoff"
    "allenai/acecoder-r1-cn-fltrd-final-ngram-filtered-chinese-filtered"
    "allenai/rlvr-code-data-python-r1-cn-fltrd-final-ngram-filtered-chinese-filtered"
    "allenai/numinatmath-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered"
)

python search.py --train_dataset_names "${usable_datasets[@]}" --ngram_size 8 --output_dir=olmo3_usable_decontam --match_threshold 0.5 --decontaminate --search_size 10000 

for dataset in "${reasoner_datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python index.py --dataset "$dataset"
done

python search.py --train_dataset_names "${reasoner_datasets[@]}" --ngram_size 8 --output_dir=olmo3_usable_decontam --match_threshold 0.5 --decontaminate --search_size 10000 
