"""
This script is used to convert the AceCoder dataset to a standard SFT format.

But it comes with a twist: we synthesize data by asking the model to generate test cases
before providing the actual implementation.

Usage: 
python scripts/data/sft/acecoder.py
python scripts/data/sft/acecoder.py --push_to_hub
python scripts/data/sft/acecoder.py --batch

# full dataset, batch, and push to hub
python scripts/data/sft/acecoder.py --full_dataset --batch --push_to_hub
"""

from dataclasses import dataclass
import json
import time
from typing import Optional
from rich.pretty import pprint

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser
from open_instruct.experimental.oai_generate import Args as OaiGenerateArgs, main as oai_generate_main

@dataclass
class Args:
    push_to_hub: bool = False
    full_dataset: bool = False
    hf_entity: Optional[str] = None
    output_jsonl: Optional[str] = None
    batch: bool = False

    def __post_init__(self):
        if self.output_jsonl is None:
            self.output_jsonl = f"output/acecoder_sft_{int(time.time())}.jsonl"
            print(f"Output file not specified, using {self.output_jsonl}")

def main(args: Args):
    dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split="train")
    if not args.full_dataset:
        dataset = dataset.select(range(5))

    def process(example):
        example["messages"] = [
            {"role": "system", "content": (
                "A conversation between User and Assistant. "
                "The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the reasoning process in "
                "the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think> </think> "
                "and <answer> </answer> tags, respectively, "
                "i.e., <think> reasoning process here </think> "
                "<answer> answer here </answer>. When given a coding question,"
                "you should think about test cases <think> tag. In the <think> tag, "
                "you should also include a ```python block that contains assert-based test cases first "
                "(without the actual implementation) and run the tests. "
                "Then, in the <answer> tag, you should include the actual implementation (without the test cases)."
            )},
            {"role": "user", "content": example["question"]},
        ]
        example["ground_truth"] = example["test_cases"]
        example["dataset"] = "ace_coder"
        return example
    
    dataset = dataset.map(process)
    # reorder columns
    dataset = dataset.select_columns(["messages", "ground_truth", "dataset"])

    # write to jsonl and generate with oai
    for idx, example in enumerate(dataset):
        data = {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": example["messages"],
                "max_tokens": 2000,
            },
        }
        with open(args.output_jsonl, "a") as f:
            f.write(json.dumps(data) + "\n")
    oai_output_jsonl = args.output_jsonl.replace(".jsonl", "_oai.jsonl")
    oai_generate_main(
        OaiGenerateArgs(
            input_file=args.output_jsonl,
            output_file=oai_output_jsonl,
            batch=args.batch,
            batch_check_interval=120,
        )
    )
    data = []
    with open(oai_output_jsonl, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Combine the data with the original dataset
    # note: oai_output_jsonl is not necessarily the same order as the dataset
    # nor does it contain all the examples
    dataset_dict = dataset.to_dict()
    print(f"{len(dataset_dict['messages'])=}")
    print(f"{len(data)=}")
    completed_idxs = set()
    for item in data:
        task_idx = int(item["custom_id"].split("-")[-1])
        completed_idxs.add(task_idx)
        try:
            dataset_dict["messages"][task_idx].append({
                "role": "assistant",
                "content": item["response"]["body"]["choices"][0]["message"]["content"]
            })
        except Exception as e:
            print(f"Error at task {task_idx}: {e}")
            pprint(item)
    dataset = datasets.Dataset.from_dict(dataset_dict)

    # Filter out the incomplete examples
    print("Filtering out the incomplete examples")
    before_dataset_len = len(dataset)
    dataset = dataset.filter(lambda x, idx: idx in completed_idxs, with_indices=True)
    after_dataset_len = len(dataset)
    print(f"Filtered {before_dataset_len - after_dataset_len} examples")

    # Filter to make sure there are 2 ```python blocks
    print("Filtering to make sure there are 2 ```python blocks")
    before_dataset_len = len(dataset)
    dataset = dataset.filter(lambda x: x["messages"][-1]["content"].count("```python") == 2)
    after_dataset_len = len(dataset)
    print(f"Filtered {before_dataset_len - after_dataset_len} examples")
    print("Example output:")

    if not args.full_dataset:
        idx = 0
        while True:
            print(dataset[idx]["messages"][-1]["content"])
            stop = input("Press Enter to continue or 'q' to quit...")
            if stop == "q":
                break
            idx += 1

    # remove the system message
    dataset = dataset.map(lambda x: {"messages": x["messages"][1:]}, num_proc=10)

    # Push to Hub
    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = HfApi().whoami()["name"]
        repo_id = f"{args.hf_entity}/acecoder_sft_gpt4o_test_cases_then_impl1"
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_type="dataset",
            repo_id=repo_id,
        )

if __name__ == "__main__":
    parser = HfArgumentParser((Args))
    main(*parser.parse_args_into_dataclasses())