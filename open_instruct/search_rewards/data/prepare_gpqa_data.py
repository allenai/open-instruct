# load the gpqa dataset from rl-rag/gpqa_diamond
from datasets import load_dataset, Dataset

# Load the original dataset
print("Loading GPQA dataset...")
ds = load_dataset("rl-rag/gpqa_diamond", split="test")
print(f"Loaded {len(ds)} examples")

# Examine the original data structure
print("\nOriginal data structure:")
print("Keys:", ds.features.keys())
if len(ds) > 0:
    print("Example:", ds[0])

# Format the data for your target format
print("\nFormatting data...")
formatted_data = []
for example in ds:
    formatted_data.append({
        "messages": [{"role": "user", "content": example["question"]}],
        "ground_truth": example["answers"],
        "dataset": "re_search"
    })

# Create new dataset from formatted data
formatted_ds = Dataset.from_list(formatted_data)
print(f"Formatted {len(formatted_ds)} examples")

# Show example of formatted data
print("\nFormatted data structure:")
print("Keys:", formatted_ds.features.keys())
if len(formatted_ds) > 0:
    print("Example:", formatted_ds[0])

# Upload to your target repository
target_repo = "rl-rag/gpqa_diamond_rlvr_no_prompt"
print(f"\nUploading to {target_repo}...")
formatted_ds.push_to_hub(target_repo, split="test")
print("Upload completed!")