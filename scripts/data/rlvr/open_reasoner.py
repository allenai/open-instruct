"""
This script is used to convert https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
Usage:

python scripts/data/rlvr/open_reasoner.py --push_to_hub
python scripts/data/rlvr/open_reasoner.py --push_to_hub --hf_entity ai2-adapt-dev
"""

import os
from collections import defaultdict
from dataclasses import dataclass

import datasets
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from transformers import HfArgumentParser


@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: str | None = None


def main(args: Args):
    # download https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/raw/refs/heads/main/data/orz_math_57k_collected.json
    import json

    import requests

    file_path = "orz_math_57k_collected.json"
    if not os.path.exists(file_path):
        url = "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/raw/refs/heads/main/data/orz_math_57k_collected.json"
        response = requests.get(url)
        with open(file_path, "w") as f:
            f.write(response.text)

    with open(file_path) as f:
        data = json.load(f)

    table = defaultdict(list)
    for item in data:
        assert len(item) == 2  # 1 question 2 ground truth
        table["messages"].append(
            [
                {"role": "user", "content": item[0]["value"]},
                {"role": "assistant", "content": item[1]["ground_truth"]["value"]},
            ]
        )
        table["ground_truth"].append(item[1]["ground_truth"]["value"])
        table["dataset"].append("math")
    dataset = datasets.Dataset.from_dict(table)

    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = HfApi().whoami()["name"]
        repo_id = f"{args.hf_entity}/rlvr_open_reasoner_math"
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=repo_id
        )

        # Add RepoCard
        repo_card = RepoCard(
            content="""\
# Open Reasoner Dataset

This dataset is converted from [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)'s math dataset.

Check out https://github.com/allenai/open-instruct/blob/main/scripts/data/rlvr/open_reasoner.py for the conversion script.

## Dataset Format

The dataset contains math problems and their solutions in a conversational format:

- `messages`: List of message dictionaries with user questions and assistant answers
- `ground_truth`: The correct solution for each problem
- `dataset`: Always "math" to indicate this is from the math datases"""
        )
        repo_card.push_to_hub(repo_id, repo_type="dataset")


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
