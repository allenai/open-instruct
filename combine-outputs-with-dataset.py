from datasets import load_dataset, Dataset

dataset = load_dataset("json", data_files="responses-from-olmo-3-base-model-100.jsonl")

def map_fn(row):
    new_messages = row["messages"]
    new_messages.append({
        "role": "assistant",
        "content": row["generated_response"]
    })
    row["messages"] = new_messages
    return row

ds = dataset.map(map_fn)
ds.push_to_hub("jacobmorrison/social-rl-eval-dataset-100")