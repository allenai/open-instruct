from datasets import load_dataset, DatasetDict

# load dataset from hub, split='test, reupload to hub with split='train'
dataset = load_dataset("allenai/reward-bench-v2-0511", split='test')

#push to hub
# push to hub with split='train'
ds = DatasetDict({'train': dataset})

# filter out for subset == 'chat:neutral'
ds = ds.filter(lambda x: x['subset'] != 'chat:neutral')
# add messages column containing [{"role": "user", "content": x['prompt']}, {"role": "assistant", "content": x['chosen']}]
ds = ds.map(lambda x: {
    "messages": [
        {"role": "user", "content": x["prompt"]}
    ]
})

ds.push_to_hub("allenai/reward-bench-v2-0511-train", private=True)
