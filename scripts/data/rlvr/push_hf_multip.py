import json
import argparse
import os
import datasets
from huggingface_hub import HfApi
from tqdm import tqdm
import re  # For extracting parts of the filename

def process_file(file_path, split_name):
    """Convert JSONL data into Hugging Face dataset format with explicit splits."""
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    dataset = {
        "messages": [[
            {"role": "user", "content": item["problem"]},
            {"role": "assistant", "content": item["answer"]},
        ] for item in data],
        "ground_truth": [item["answer"] for item in data],
        "dataset": ["multiplication"] * len(data),
        "split": [split_name] * len(data),
    }
    return datasets.Dataset.from_dict(dataset)

def push_to_huggingface(saved_files, hf_entity):
    """Push each multiplication dataset (e.g., `10x10_train_1000`, `2x2_test_100`) separately to Hugging Face."""
    api = HfApi()
    hf_entity = hf_entity or api.whoami()["name"]

    print("\n📤 Uploading selected multiplication datasets to Hugging Face...\n")

    for name, files in tqdm(saved_files.items(), desc="Uploading datasets"):
        train_data = process_file(files["train"], "train") if "train" in files else None
        test_data = process_file(files["test"], "test") if "test" in files else None

        dataset_dict = datasets.DatasetDict()
        if train_data:
            dataset_dict["train"] = train_data
        if test_data:
            dataset_dict["train"] = test_data

        dimensions, split, size = name.split("_")  # Corrected order

        # Now correctly formatted repo name
        repo_id = f"{hf_entity}/multiplication_{split}_{size}_{dimensions}"

        # Only push if at least one dataset exists
        if dataset_dict:
            dataset_dict.push_to_hub(repo_id)
            print(f"✅ Dataset `{name}` uploaded successfully to {repo_id}")
        else:
            print(f"⚠️ Skipping `{name}`: No train/test data found.")

def extract_dataset_info(filename):
    """
    Extract dataset name, size, and split from filename.
    Example: "multiplication_1000_train_10x10.jsonl" → "10x10_train_1000"
    """
    match = re.match(r"multiplication_(\d+)_(train|test)_(\d+x\d+)\.jsonl", filename)
    if match:
        size, split, dimensions = match.groups()
        dataset_name = f"{dimensions}_{split}_{size}"  # Example: "10x10_train_1000"
        return dataset_name, split
    return None, None  # If it doesn't match expected pattern

def read_selected_datasets(input_dir, train_files, test_files):
    """Prepare a dictionary of specific train and test files with their paths."""
    saved_files = {}
    for filename in train_files + test_files:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            dataset_name, split = extract_dataset_info(filename)
            if dataset_name and split:
                if dataset_name not in saved_files:
                    saved_files[dataset_name] = {}
                saved_files[dataset_name][split] = file_path
        else:
            print(f"⚠️ Warning: {file_path} not found.")

    return saved_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/weka/oe-adapt-default/nouhad/data/math_data/multiplication", help="Directory containing JSONL files")
    parser.add_argument("--hf_entity", type=str, default=None, help="Hugging Face entity")
    args = parser.parse_args()

    input_dir = args.input_dir
    train_files = ["multiplication_1000_train_10x10.jsonl"]
    train_files = []
    # test_files = ["multiplication_100_test_2x2.jsonl"]
    test_files = ["multiplication_100_test_2x2.jsonl", "multiplication_100_test_6x6.jsonl", "multiplication_100_test_12x12.jsonl"]


    # Read only the selected files
    saved_files = read_selected_datasets(input_dir, train_files, test_files)

    # Push datasets to Hugging Face
    push_to_huggingface(saved_files, args.hf_entity)

if __name__ == "__main__":
    main()
