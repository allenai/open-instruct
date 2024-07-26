
# preparation 

Make sure you are in the `rejection_sampling` folder.


# Debug run

```bash
# 1. first sample a bunch of completions given prompts
python generation.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --sanity_check \

# 2. tokenize them and run a reward model to filter them
python rejection_sampling.py \
    --input_filename completions.jsonl \
    --save_filename rejection_sampled_completions.jsonl \
    --n 3 \
    --num_gpus 2 \
    --push_to_hub
```


# Run through the entire dataset run



```bash
# 1. first sample a bunch of completions given prompts
python generation.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --n 3 \
    --tensor_parallel_size 8 \

# 2. tokenize them and run a reward model to filter them
python rejection_sampling.py \
    --input_filename completions.jsonl \
    --save_filename rejection_sampled_completions.jsonl \
    --n 3 \
    --num_gpus 2 \
    --push_to_hub
```

# 3. run the SFT loss on the best ones
