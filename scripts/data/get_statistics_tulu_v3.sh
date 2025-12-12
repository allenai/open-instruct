#!/bin/bash
# Run with scripts/data/get_statistics_tulu_v3.sh

# List of datasets
datasets=(
    ai2-adapt-dev/oasst1_converted
    ai2-adapt-dev/flan_v2_converted
    ai2-adapt-dev/tulu_hard_coded_repeated_10
    ai2-adapt-dev/no_robots_converted
    ai2-adapt-dev/tulu_v3.9_wildchat_100k
    ai2-adapt-dev/personahub_math_v5_regen_149960
    allenai/tulu-3-sft-personas-math-grade
    ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k
    ai2-adapt-dev/numinamath_tir_math_decontaminated
    ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k
    ai2-adapt-dev/personahub_code_v2_34999
    ai2-adapt-dev/evol_codealpaca_heval_decontaminated
    ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980
    ai2-adapt-dev/coconot_converted
    ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k
    ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k
    ai2-adapt-dev/tulu_v3.9_sciriff_10k
    ai2-adapt-dev/tulu_v3.9_table_gpt_5k
    ai2-adapt-dev/tulu_v3.9_aya_100k
)

# For every dataset, get the statistics if the output directory doesn't exist
for dataset in "${datasets[@]}"; do
    output_file="data/processed/${dataset}_statistics.json"

    if [ ! -f "$output_file" ]; then
        echo "Getting statistics for $dataset..."
        python scripts/data/get_statistics.py --data_path ${dataset} --save_path ${output_file}
    else
        echo "Statistics for $dataset already exist. Skipping..."
    fi
done

# list of preference datasets
datasets_pref=(
    allenai/ultrafeedback_binarized_cleaned_train
    ai2-adapt-dev/DaringAnteater-prefs-RM-filter
    ai2-adapt-dev/WildChat-prefs-280824
    ai2-adapt-dev/Llama-3.1-if_taxonomy_tulu
)

# For every dataset, get the statistics if the output directory doesn't exist
for dataset in "${datasets_pref[@]}"; do
    output_file="data/processed/${dataset}_statistics.json"

    if [ ! -f "$output_file" ]; then
        echo "Getting statistics for $dataset..."
        python scripts/data/get_statistics.py --data_path ${dataset} --save_path ${output_file} --messages_key chosen
    else
        echo "Statistics for $dataset already exist. Skipping..."
    fi
done
