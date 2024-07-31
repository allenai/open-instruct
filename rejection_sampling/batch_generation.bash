mkdir -p rejection_sampling/shards
total_items=326154
num_shards=10
items_per_shard=$((total_items / num_shards))

# Loop through shards
for ((i=0; i<num_shards; i++))
do
    # Calculate start and end indices for this shard
    start_idx=$((i * items_per_shard))
    end_idx=$(((i + 1) * items_per_shard))
    
    # Adjust the end index for the last shard to include any remaining items
    if [ $i -eq $((num_shards - 1)) ]; then
        end_idx=$((total_items))
    fi
    
    # Run the command for this shard
    echo "Running shard $((i+1)) of $num_shards (indices $start_idx to $end_idx)"
    python mason.py \
        --cluster ai2/allennlp-cirrascale  ai2/general-cirrascale-a5000  ai2/general-cirrascale-a5000  ai2/general-cirrascale-a100-80g-ib  \
        --budget ai2/allennlp \
        --priority low \
        --gpus 1 -- python rejection_sampling/generation.py \
        --dataset_name allenai/tulu-v2-sft-mixture \
        --model_name_or_path allenai/llama-3-tulu-2-8b \
        --dataset_start_idx $start_idx \
        --dataset_end_idx $end_idx \
        --save_filename rejection_sampling/shards/rejection_sampled_completions_$i.jsonl \
        --n 5

    echo "Finished shard $((i+1)) of $num_shards"
    echo
done
echo "All shards submitted"