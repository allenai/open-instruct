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
                    "Answer the given question. You must conduct reasoning inside <think> and </think> "
                    "first every time you get new information. After reasoning, if you find you lack some "
                    "knowledge, you can call a search engine by <query> query </query>, and it will return "
                    "the top searched results between <output> and </output>. You can search as many "
                    "times as you want. If you find no further external knowledge needed, you can directly "
                    "provide the answer inside <finish> and </finish> without detailed illustrations. "
                    "For example, <finish> xxx </finish>. Question: "
                    f"{data['problem']}"
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
                    "Answer the given question. You must conduct reasoning inside <think> and </think> "
                    "first every time you get new information. After reasoning, if you find you lack some "
                    "knowledge, you can call a search engine by <query> query </query>, and it will return "
                    "the top searched results between <output> and </output>. You can search as many "
                    "times as you want. If you find no further external knowledge needed, you can directly "
                    "provide the answer inside <finish> and </finish> without detailed illustrations. "
                    "For example, <finish> xxx </finish>. Question: "
                    f"{data['problem']}"
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
