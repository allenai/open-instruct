#!/bin/zsh

datasets=(
    "allenai/tulu-3-sft-personas-instruction-following"
    "allenai/tulu-3-sft-personas-math"
    "allenai/tulu-3-sft-personas-math-grade"
    "allenai/tulu-3-sft-personas-algebra"
)
ids=(
    "batch_673e3942-e8d2-4d4a-bfd5-dc88deacff65"
    "batch_b4893a3b-28fd-47f4-86cd-9a9c2e6393b0,batch_5764a197-2782-4dac-a5ef-f39fb86aaa56"
    "batch_546866f6-dff4-416d-81ef-1c172e956135"
    "batch_2b418785-35a4-4ba8-b4d1-d3e5d0d46e03"
)

for i in {1..${#datasets}}; do
    dataset="${datasets[$i]}"
    batch_ids="${ids[$i]}"
    
    # Split batch_ids into array
    batch_array=(${=batch_ids//,/ })
    
    for batch_id in "${batch_array[@]}"; do
        output_dataset=$(echo "$dataset" | sed 's/allenai/finbarr/')-o3
        python process_azure_batch_results.py \
          "$batch_id" \
          --input-dataset "$dataset" \
          --output-dataset "$output_dataset"
        echo "Processed batch $batch_id for dataset $dataset to $output_dataset"
    done
done
