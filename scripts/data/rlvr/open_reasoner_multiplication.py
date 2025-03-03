"""
This script processes all multiplication JSONL files in:
    /weka/oe-adapt-default/nouhad/data/multiplication/

Usage:
    python scripts/data/rlvr/open_reasoner.py --push_to_hub
    python scripts/data/rlvr/open_reasoner.py --push_to_hub --hf_entity ai2-adapt-dev
"""

from collections import defaultdict
from dataclasses import dataclass
import os
import json
from typing import Optional

import datasets
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from transformers import HfArgumentParser

@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: Optional[str] = None

def process_file(file_path: str):
    """
    Processes a single JSONL file into a dataset.
    """
    print(f"Processing file: {file_path}")
    
    with open(file_path, "r") as f:
        data = f.readlines()

    new_data = []
    for item in data:
        item = item.strip()  # Remove extra spaces or newline characters
        if not item:  # Skip empty lines
            continue
        new_data.append(json.loads(item))  # Convert JSON string to Python dict

    table = defaultdict(list)
    for item in new_data:
        assert "problem" in item and "answer" in item, "Missing expected keys in data"
        table["messages"].append([
            {"role": "user", "content": item["problem"]},
            {"role": "assistant", "content": item["answer"]},
        ])
        table["ground_truth"].append(item["answer"])
        table["dataset"].append("multiplication")

    return datasets.Dataset.from_dict(table)

def main(args: Args):
    data_dir = "/weka/oe-adapt-default/nouhad/data/multiplication/"
    jsonl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    api = HfApi()
    hf_entity = args.hf_entity if args.hf_entity else api.whoami()["name"]

    for file_path in jsonl_files:
        dataset = process_file(file_path)
        dataset_name = os.path.basename(file_path).replace(".jsonl", "")

        if args.push_to_hub:
            repo_id = f"{hf_entity}/{dataset_name}"
            print(f"Pushing dataset to Hub: {repo_id}")

            dataset.push_to_hub(repo_id)
            api.upload_file(
                path_or_fileobj=__file__,
                path_in_repo="create_dataset.py",
                repo_type="dataset",
                repo_id=repo_id,
            )

            # Add RepoCard
            repo_card = RepoCard(
                content=f"""\
# Multiplication Dataset - {dataset_name}

This dataset contains multiplication problems and their solutions.

## Dataset Format

- `messages`: List of message dictionaries with user questions and assistant answers
- `ground_truth`: The correct solution for each problem
- `dataset`: Always "multiplication" to indicate its type
"""
            )
            repo_card.push_to_hub(
                repo_id,
                repo_type="dataset",
            )

if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    main(*parser.parse_args_into_dataclasses())
