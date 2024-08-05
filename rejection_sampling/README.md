
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
    --gpus 1 -- which python
    
chmod -R 777 /net/nfs.cirrascale/allennlp/.cache/hub/
python mason.py \
    --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a100-80g-ib \
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
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '0', '--dataset_end_idx', '32615', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_0.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJ6GFY327Q3GMH81T7X9J
# Finished shard 1 of 10

# Running shard 2 of 10 (indices 32615 to 65230)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '32615', '--dataset_end_idx', '65230', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_1.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJ797SXMFNPCNNFF278Z5
# Finished shard 2 of 10

# Running shard 3 of 10 (indices 65230 to 97845)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '65230', '--dataset_end_idx', '97845', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_2.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJ814RVA87G38YG949NK5
# Finished shard 3 of 10

# Running shard 4 of 10 (indices 97845 to 130460)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '97845', '--dataset_end_idx', '130460', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_3.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJ8SAV2R1ZYBVZM1X4VP0
# Finished shard 4 of 10

# Running shard 5 of 10 (indices 130460 to 163075)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '130460', '--dataset_end_idx', '163075', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_4.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJ9HHEMZRREPKSBENWJB1
# Finished shard 5 of 10

# Running shard 6 of 10 (indices 163075 to 195690)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '163075', '--dataset_end_idx', '195690', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_5.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJA9B3GHJ3CTPATTV1342
# Finished shard 6 of 10

# Running shard 7 of 10 (indices 195690 to 228305)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '195690', '--dataset_end_idx', '228305', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_6.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJB1Y6KY564923RJRK39B
# Finished shard 7 of 10

# Running shard 8 of 10 (indices 228305 to 260920)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '228305', '--dataset_end_idx', '260920', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_7.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJBSTVEXV8WJM6R09R9CT
# Finished shard 8 of 10

# Running shard 9 of 10 (indices 260920 to 293535)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '260920', '--dataset_end_idx', '293535', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_8.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJCJ5HEVVB0TMSVKFEBAQ
# Finished shard 9 of 10

# Running shard 10 of 10 (indices 293535 to 326154)
# full_command=['python', 'rejection_sampling/generation.py', '--dataset_name', 'allenai/tulu-v2-sft-mixture', '--model_name_or_path', 'allenai/llama-3-tulu-2-8b', '--dataset_start_idx', '293535', '--dataset_end_idx', '326154', '--save_filename', 'rejection_sampling/shards/rejection_sampled_completions_9.jsonl', '--n', '5']
# Kicked off Beaker job. https://beaker.org/ex/01J45EJDAAKXEW084RF71QYMFP
# Finished shard 10 of 10

# All shards submitted


bash rejection_sampling/batch_rejection_sampling.bash
```



```
huggingface-cli upload vwxyzjn/rejection_sampling_23251 . .
```