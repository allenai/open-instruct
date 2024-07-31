
# preparation 

Make sure you are in the `rejection_sampling` folder.


# Debug run (use an interactive session)

```bash
## tulu v3 recipe
# 1. first sample a bunch of completions given prompts
python rejection_sampling/generation.py \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --model_name_or_path allenai/llama-3-tulu-2-8b \
    --n 3 \
    --save_filename rejection_sampling/completions.jsonl \
    --sanity_check \
    
# 2. tokenize them and run a reward model to filter them
python rejection_sampling/rejection_sampling.py \
    --input_filename rejection_sampling/completions.jsonl \
    --model_name_or_path allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --save_filename rejection_sampling/rejection_sampled_completions.jsonl \
    --n 3 \
    --push_to_hub \
    --num_gpus 1 \
```



# Run through the entire dataset run

To run through the entire dataset you would need a lot more GPUs to finish the generation more quickly. 

```bash
# debug job submission
python mason.py \
    --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/general-cirrascale-a100-80g-ib \
    --priority low \
    --budget ai2/allennlp \
    --gpus 1 -- python rejection_sampling/generation.py \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --model_name_or_path allenai/llama-3-tulu-2-8b \
    --dataset_start_idx 0 \
    --n 5 \
    --sanity_check

# prod generations
bash rejection_sampling/batch_generation.bash
# Running shard 1 of 10 (indices 0 to 32615)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '0', '--dataset_end_idx', '32615', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_0.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6E9N8PQBAHHRGP3YWY2A
# Finished shard 1 of 10

# Running shard 2 of 10 (indices 32615 to 65230)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '32615', '--dataset_end_idx', '65230', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_1.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6F26QK71XEB5SX65JZWF
# Finished shard 2 of 10

# Running shard 3 of 10 (indices 65230 to 97845)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '65230', '--dataset_end_idx', '97845', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_2.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6FTYQYACTQETVKQ30Q26
# Finished shard 3 of 10

# Running shard 4 of 10 (indices 97845 to 130460)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '97845', '--dataset_end_idx', '130460', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_3.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6GKH6H5JMBH71T9DHHRB
# Finished shard 4 of 10

# Running shard 5 of 10 (indices 130460 to 163075)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '130460', '--dataset_end_idx', '163075', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_4.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6HC6TZ545TBK9HFEWB8Z
# Finished shard 5 of 10

# Running shard 6 of 10 (indices 163075 to 195690)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '163075', '--dataset_end_idx', '195690', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_5.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6J47T8769T3ES13JYK3Q
# Finished shard 6 of 10

# Running shard 7 of 10 (indices 195690 to 228305)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '195690', '--dataset_end_idx', '228305', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_6.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6JWME19A0Z3BZ1N65HB5
# Finished shard 7 of 10

# Running shard 8 of 10 (indices 228305 to 260920)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '228305', '--dataset_end_idx', '260920', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_7.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6KMW3Z10E5546CHXJZC3
# Finished shard 8 of 10

# Running shard 9 of 10 (indices 260920 to 293535)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '260920', '--dataset_end_idx', '293535', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_8.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6MDJD0DW97ZTM6R1PE03
# Finished shard 9 of 10

# Running shard 10 of 10 (indices 293535 to 326154)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '293535', '--dataset_end_idx', '326154', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_9.jsonl', '--n', '5', '--priority', 'low']
# Kicked off Beaker job. https://beaker.org/ex/01J45E6N5P6C9TSAX1PGS1635K
# Finished shard 10 of 10
```


# 2. tokenize them and run a reward model to filter them
python rejection_sampling.py \
    --input_filename completions.jsonl \
    --save_filename rejection_sampled_completions.jsonl \
    --n 3 \
    --num_gpus 2 \
    --push_to_hub
```



# build docker 
