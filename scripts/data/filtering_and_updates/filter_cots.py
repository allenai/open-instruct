#!/usr/bin/env python
"""Quick script to filter a dataset of reasoner model generations for proper thinking
token format and answer token format.

Example usage for Olmo 3 dataset filtering:
python filter_cots.py --input_dataset_name allenai/oasst1-r1 --output_dataset_name allenai/oasst1-r1-format-filtered --filter think --output_format no_answer
"""

import argparse
import re

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils


# ----------------------- filter functions ----------------------- #
def is_think_answer(elem):
    """
    Return True if the string has the exact form
    <think>...</think><answer>...</answer>, False otherwise.
    Also returns false if 'messages' or last message's 'content' is None
    """
    pattern = re.compile(r"^<think>[\s\S]*?</think><answer>[\s\S]*?</answer>$")
    try:
        return bool(pattern.fullmatch(elem["messages"][-1]["content"]))
    except Exception:
        print("KeyError: 'messages' or 'content' not found in element")
        return False


def is_think(elem):
    """
    Return True if the string has the exact form
    <think>...</think>, False otherwise.
    Also returns false if 'messages' or last message's 'content' is None
    """
    pattern = re.compile(r"^<think>[\s\S]*?</think>[\s\S]*?$")
    try:
        return bool(pattern.fullmatch(elem["messages"][-1]["content"]))
    except Exception:
        print("KeyError: 'messages' or 'content' not found in element")
        return False


# ----------------------- map functions ----------------------- #
def add_answer(elem):
    """
    Modifies the last assistant turn to include <answer></answer>
    tags if they do not already exist
    """
    assert elem["messages"][-1]["role"] == "assistant", "Expected the last message to be from the assistant."
    if "</think>\n" not in elem["messages"][-1]["content"]:
        elem["messages"][-1]["content"] = elem["messages"][-1]["content"].replace("</think>", "</think>\n")
    if "<answer>" not in elem["messages"][-1]["content"]:
        elem["messages"][-1]["content"] = elem["messages"][-1]["content"].replace("</think>", "</think>\n<answer>")
    if not elem["messages"][-1]["content"].endswith("</answer>"):
        elem["messages"][-1]["content"] = elem["messages"][-1]["content"] + "\n</answer>"
    return elem


def remove_answer(elem):
    """
    Modifies the last assistant turn to not include
    <answer></answer> tags if they exist
    """
    assert elem["messages"][-1]["role"] == "assistant", "Expected the last message to be from the assistant."
    elem["messages"][-1]["content"] = (
        elem["messages"][-1]["content"].replace("<answer>", "").replace("</answer>", "").strip()
    )
    if "</think>\n" not in elem["messages"][-1]["content"]:
        elem["messages"][-1]["content"] = elem["messages"][-1]["content"].replace("</think>", "</think>\n")
    return elem


# ----------------------- main script ----------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a Hugging Face dataset by message tag pattern and push the result to the Hub."
    )
    parser.add_argument(
        "-i",
        "--input_dataset_name",
        required=True,
        help="Name or path of the input dataset (e.g. 'username/dataset_name').",
    )
    parser.add_argument(
        "-o", "--output_dataset_name", required=True, help="Name of the destination dataset repo on the Hub."
    )
    parser.add_argument(
        "-f",
        "--filter",
        choices=["think_answer", "think"],
        default="think",
        help="Which filter function to apply (default: think).",
    )
    parser.add_argument(
        "--output_format",
        choices=["answer", "keep", "no_answer"],
        default="no_answer",
        help="Whether to include <answer></answer> tags in the final model response (default: no_answer).",
    )
    args = parser.parse_args()

    # Load dataset (Dataset or DatasetDict)
    ds = load_dataset(args.input_dataset_name, num_proc=open_instruct_utils.max_num_processes())

    # Helper: count rows across splits
    def count_rows(dset):
        if hasattr(dset, "values"):  # DatasetDict
            return sum(len(split) for split in dset.values())
        return len(dset)

    # Select filter function
    filter_fn = is_think_answer if args.filter == "think_answer" else is_think

    # Keep rows that satisfy the predicate
    kept_ds = ds.filter(filter_fn)
    if args.output_format == "answer":
        kept_ds = kept_ds.map(add_answer)
    elif args.output_format == "keep":
        pass
    else:
        kept_ds = kept_ds.map(remove_answer)

    # Keep rows that FAIL the predicate (invert the bool)
    filtered_out_ds = ds.filter(lambda elem: not filter_fn(elem))

    # Stats
    original_n = count_rows(ds)
    kept_n = count_rows(kept_ds)
    filtered_out_n = count_rows(filtered_out_ds)

    print(f"Original dataset       : {original_n:,} examples")
    print(f"Kept after filtering   : {kept_n:,} examples ({kept_n/original_n:.2%})")
    print(f"Filtered-out examples  : {filtered_out_n:,} examples ({filtered_out_n/original_n:.2%})")

    # Push both datasets
    kept_ds.push_to_hub(args.output_dataset_name, private=True)
    filtered_repo = f"{args.output_dataset_name}-filtered"
    filtered_out_ds.push_to_hub(filtered_repo, private=True)
    print(f"Pushed kept rows to      : {args.output_dataset_name}")
    print(f"Pushed filtered-out rows : {filtered_repo}")


if __name__ == "__main__":
    main()
