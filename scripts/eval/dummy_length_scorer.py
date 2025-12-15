"""
Dummy evaluator that uses a given metric to determine winners in pairwise comparisons. Used to further investigate correlations.
"""

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer

import open_instruct.utils as open_instruct_utils

parser = argparse.ArgumentParser()
parser.add_argument("--candidate_file", type=str, help="Candidate file for candidate model outputs.")
parser.add_argument("--metric", default="unique", type=str, help="Metric to use for comparison.")
parser.add_argument(
    "--tokenizer",
    default="/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B",
    type=str,
    help="Tokenizer to use for tokenization.",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)


def count_unique_tokens(text):
    return len(set(tokenizer(text).input_ids))


def count_token_length(text):
    return len(tokenizer(text).input_ids)


metric_map = {"unique": count_unique_tokens, "length": count_token_length}

if __name__ == "__main__":
    # load reference data
    reference_dataset = load_dataset(
        "hamishivi/alpaca-farm-davinci-003-2048-token", num_proc=open_instruct_utils.max_num_processes()
    )
    reference_dataset = [x["output"] for x in reference_dataset["train"]]
    # load candidate data
    with open(args.candidate_file) as f:
        candidate_dataset = json.load(f)
        candidate_dataset = [x["output"] for x in candidate_dataset]
    win_counter = 0
    lose_counter = 0
    tie_counter = 0
    # compute metrics - we assume same order of reference and candidate data
    for reference_sample, candidate_sample in zip(reference_dataset, candidate_dataset):
        reference_metric = metric_map[args.metric](reference_sample)
        candidate_metric = metric_map[args.metric](candidate_sample)
        if reference_metric > candidate_metric:
            lose_counter += 1
        elif reference_metric < candidate_metric:
            win_counter += 1
        else:
            tie_counter += 1

    print(f"{win_counter}\t{lose_counter}\t{tie_counter}")
