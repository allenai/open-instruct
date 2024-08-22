#!/bin/bash

mkdir -p output/shards
num_prompts=1000
num_shards=4
prompts_per_shard=$((num_prompts / num_shards))
timestamp=$RANDOM
shared_generation_hf_repo_id=generation_$timestamp
shared_rs_hf_repo_id=rejection_sampling_$timestamp
shared_scores_hf_repo_id=scores_$timestamp
num_completions=5
generation_model=allenai/llama-3-tulu-2-8b
reward_model=allenai/llama-3-tulu-2-8b-uf-mean-rm
sft_dataset=allenai/tulu-v2-sft-mixture
num_gpus=1
mkdir -p output/shards/$timestamp

# Prepare the command string
command=""

# Loop through shards
for ((i=0; i<num_shards; i++))
do
    # Calculate start and end indices for this shard
    start_idx=$((i * prompts_per_shard))
    end_idx=$(((i + 1) * prompts_per_shard))
    
    # Adjust the end index for the last shard to include any remaining prompts
    if [ $i -eq $((num_shards - 1)) ]; then
        end_idx=$num_prompts
    fi
    
    # Build the command string for this shard
    shard_command="python open_instruct/rejection_sampling/generation.py \
    --dataset_name $sft_dataset \
    --model_name_or_path $generation_model \
    --dataset_start_idx $start_idx \
    --dataset_end_idx $end_idx \
    --save_filename output/shards/$timestamp/$i.jsonl \
    --hf_repo_id $shared_generation_hf_repo_id \
    --no_add_timestamp \
    --push_to_hub \
    --num_completions $num_completions --tensor_parallel_size $num_gpus && \
    python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/shards/$timestamp/$i.jsonl \
    --model_names_or_paths $reward_model \
    --save_filename output/shards/$timestamp/rs_$i.jsonl \
    --save_filename_scores output/shards/$timestamp/scores_$i.jsonl \
    --hf_repo_id $shared_rs_hf_repo_id \
    --hf_repo_id_scores $shared_scores_hf_repo_id \
    --no_add_timestamp \
    --num_completions $num_completions \
    --push_to_hub \
    --num_gpus $num_gpus && \
    echo Finished shard $((i+1)) of $num_shards"

    # Add the shard command to the main command string
    if [ -z "$command" ]; then
        command="$shard_command"
    else
        command="$command -- $shard_command"
    fi
done

echo $command

# Run the combined command
echo "Submitting all shards in one command"
python mason.py \
    --cluster ai2/general-cirrascale-a5000 ai2/allennlp-cirrascale ai2/s2-cirrascale ai2/mosaic-cirrascale \
    --priority low \
    --preemptible \
    --budget ai2/allennlp \
    --gpus $num_gpus -- $command

echo "All shards submitted"