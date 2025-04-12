from datasets import load_dataset, DatasetDict, Dataset

# load dataset
dataset = load_dataset("basicv8vc/SimpleQA", split="test")
# train-test split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
# construct rlvr-style dataset
new_train_data = []
for data in dataset['train']:
    new_train_data.append({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{data['problem']} "
                    "Search the web by wrapping a query in query tags like so: <query></query> "
                    "Then, based on the snippet, provide the answer, or another query if you need. "
                    "Finally, output your answer wrapped in answer tags: <finish></finish>."
                )
            },
        ],
        "ground_truth": data['answer'],
        "dataset": "re_search"
    })

new_test_data = []
for data in dataset['test']:
    new_test_data.append({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{data['problem']} "
                    "Search the web by wrapping a query in query tags like so: <query></query> "
                    "Then, based on the snippet, provide the answer, or another query if you need. "
                    "Finally, output your answer wrapped in answer tags: <finish></finish>."
                )
            },
        ],
        "ground_truth": data['answer'],
        "dataset": "re_search"
    })

ds = DatasetDict({
    "train": Dataset.from_list(new_train_data),
    "test": Dataset.from_list(new_test_data)
})
ds = ds.shuffle(seed=42)
ds.push_to_hub("hamishivi/SimpleQA-RLVR")

# create no-prompt simpleqa dataset
# construct rlvr-style dataset
new_train_data = []
for data in dataset['train']:
    new_train_data.append({
        "messages": [
            {
                "role": "user",
                "content": data['problem']
            },
        ],
        "ground_truth": data['answer'],
        "dataset": "re_search"
    })

new_test_data = []
for data in dataset['test']:
    new_test_data.append({
        "messages": [
            {
                "role": "user",
                "content": data['problem']
            },
        ],
        "ground_truth": data['answer'],
        "dataset": "re_search"
    })

ds = DatasetDict({
    "train": Dataset.from_list(new_train_data),
    "test": Dataset.from_list(new_test_data)
})
ds = ds.shuffle(seed=42)
ds.push_to_hub("hamishivi/SimpleQA-RLVR-noprompt")
