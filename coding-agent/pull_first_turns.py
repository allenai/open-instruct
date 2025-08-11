from datasets import load_dataset, Dataset

INPUT_DS = "saurabh5/rlvr-code-view-tool"
OUTPUT_DS = "saurabh5/rlvr-code-view-tool-queries"

ds = load_dataset(INPUT_DS, split="train")
new_ds = []

for row in ds:
    new_rows = {**row}
    new_rows["messages"] = row["messages"][:2]
    new_ds.append(new_rows)

ds = Dataset.from_list(new_ds)
ds.push_to_hub(OUTPUT_DS)