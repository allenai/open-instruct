mkdir -p output/shards
num_prompts=400
num_shards=5
prompts_per_shard=$((num_prompts / num_shards))
shared_hf_repo_id=rejection_sampling_$RANDOM 
num_generations=5
generation_model=allenai/llama-3-tulu-2-8b
reward_model=allenai/llama-3-tulu-2-8b-uf-mean-rm
sft_dataset=allenai/tulu-v2-sft-mixture
num_gpus=1
clusters=ai2/general-cirrascale-a5000 # ai2/allennlp-cirrascale ai2/general-cirrascale-a100-80g-ib
mkdir -p output/shards/$shared_hf_repo_id

# Loop through shards
for ((i=0; i<num_shards; i++))
do
    # Calculate start and end indices for this shard
    start_idx=$((i * prompts_per_shard))
    end_idx=$(((i + 1) * prompts_per_shard))
    
    # Adjust the end index for the last shard to include any remaining prompts
    if [ $i -eq $((num_shards - 1)) ]; then
        end_idx=$((num_prompts))
    fi
    
    # Run the command for this shard
    echo "Running shard $((i+1)) of $num_shards (indices $start_idx to $end_idx)"
    python mason.py \
        --cluster $clusters  \
        --budget ai2/allennlp \
        --priority low \
        --gpus $num_gpus -- "python open_instruct/generation.py \
        --dataset_name $sft_dataset \
        --model_name_or_path $generation_model \
        --dataset_start_idx $start_idx \
        --dataset_end_idx $end_idx \
        --save_filename output/shards/$shared_hf_repo_id/$i.jsonl \
        --n $num_generations --tensor_parallel_size $num_gpus && python open_instruct/rejection_sampling.py \
        --input_filename output/shards/$shared_hf_repo_id/$i.jsonl \
        --model_name_or_path $reward_model \
        --save_filename output/shards/$shared_hf_repo_id/scores_$i.jsonl \
        --hf_repo_id $shared_hf_repo_id \
        --no_add_timestamp \
        --n $num_generations \
        --push_to_hub \
        --num_gpus $num_gpus"

    echo "Finished shard $((i+1)) of $num_shards"
    echo
done
echo "All shards submitted"