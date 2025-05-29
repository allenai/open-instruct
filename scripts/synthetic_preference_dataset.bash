#!/bin/bash

# Default values with override option
num_prompts=${num_prompts:-9500}
num_shards=${num_shards:-4}
num_completions=${num_completions:-3}
generation_model=${generation_model:-"allenai/open_instruct_dev"}
generation_model_revision=${generation_model_revision:-"costa_finetune_tulu3_8b_norobot__meta-llama_Meta-Llama-3.1-8B__42__1725559869"}
judge_model=${judge_model:-"gpt-4o-2024-08-06"}
dataset_mixer_list=${dataset_mixer_list:-"HuggingFaceH4/no_robots 1.0"}
dataset_splits=${dataset_splits:-"train"}
deploy_mode=${deploy_mode:-"nfs"}
num_gpus=${num_gpus:-1}
exp_name=${exp_name:-"norobot"}
timestamp=${timestamp:-$RANDOM}

# Derived values
prompts_per_shard=$((num_prompts / num_shards))
shared_generation_hf_repo_id="${exp_name}_generation_${timestamp}"
shared_synthetic_pref_hf_repo_id="${exp_name}_pref_${timestamp}"

# Print the values (for debugging or informational purposes)
echo "Number of prompts: $num_prompts"
echo "Number of shards: $num_shards"
echo "Prompts per shard: $prompts_per_shard"
echo "Number of completions: $num_completions"
echo "Generation model: $generation_model"
echo "Generation model revision: $generation_model_revision"
echo "Judge model: $judge_model"
echo "Dataset mixer list: $dataset_mixer_list"
echo "Deploy mode: $deploy_mode"
echo "Number of GPUs: $num_gpus"
echo "Shared generation HF repo ID: $shared_generation_hf_repo_id"
echo "Shared synthetic pref HF repo ID: $shared_synthetic_pref_hf_repo_id"

# Create output directory
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
    --dataset_mixer_list $dataset_mixer_list \
    --dataset_splits $dataset_splits \
    --model_name_or_path $generation_model \
    --revision $generation_model_revision \
    --dataset_start_idx $start_idx \
    --dataset_end_idx $end_idx \
    --save_filename /output/shards/$timestamp/$i.jsonl \
    --hf_repo_id $shared_generation_hf_repo_id \
    --no_add_timestamp \
    --push_to_hub \
    --num_completions $num_completions --tensor_parallel_size $num_gpus && \
    python open_instruct/rejection_sampling/synthetic_preference_dataset.py \
    --input_filename /output/shards/$timestamp/$i.jsonl \
    --model $judge_model \
    --save_filename /output/shards/$timestamp/synth_$i.jsonl \
    --hf_repo_id $shared_synthetic_pref_hf_repo_id \
    --no_add_timestamp \
    --num_completions $num_completions\
    --push_to_hub && echo Finished shard 1 of 4
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
if [ "$deploy_mode" = "docker_weka" ]; then
    python mason.py \
        --cluster ai2/neptune-cirrascale ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 \
        --image costah/open_instruct_synth_pref --pure_docker_mode \
        --priority low \
        --preemptible \
        --budget ai2/allennlp \
        --gpus $num_gpus -- $command
elif [ "$deploy_mode" = "docker_nfs" ]; then
    python mason.py \
        --cluster ai2/allennlp-cirrascale ai2/pluto-cirrascale ai2/s2-cirrascale \
        --image costah/open_instruct_synth_pref --pure_docker_mode \
        --priority low \
        --preemptible \
        --budget ai2/allennlp \
        --gpus $num_gpus -- $command
elif [ "$deploy_mode" = "docker" ]; then
    python mason.py \
        --cluster ai2/allennlp-cirrascale ai2/neptune-cirrascale \
        --image costah/open_instruct_synth_pref --pure_docker_mode \
        --priority low \
        --preemptible \
        --budget ai2/allennlp \
        --gpus $num_gpus -- $command
elif [ "$deploy_mode" = "nfs" ]; then
    python mason.py \
        --cluster ai2/allennlp-cirrascale ai2/pluto-cirrascale ai2/s2-cirrascale \
        --priority low \
        --preemptible \
        --budget ai2/allennlp \
        --gpus $num_gpus -- $command
else
    echo "Invalid deploy_mode"
fi
echo "All shards submitted"