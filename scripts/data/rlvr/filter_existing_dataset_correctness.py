#!/usr/bin/env python3
"""
Filter an existing jsonl dataset by correctness, using multiple processes.
requires reward functions setup.
we use multiprocessing to make things actually fast.

to run:
python scripts/data/rlvr/filter_existing_dataset_correctness.py \
  --files data/*.jsonl --output_file filtered.jsonl

If you have code data, you might have to launch code server too before running:
source configs/beaker_configs/code_api_setup.sh
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

import matplotlib.pyplot as plt
from tqdm import tqdm

from open_instruct.ground_truth_utils import build_all_verifiers


def _avg_correctness(sample, reward_fn_mapping):
    """
    Compute the mean correctness for one sample (called in worker).
    """
    dataset = sample["dataset"][0]
    gt = sample["ground_truth"][0]
    outputs = sample["output"]

    reward_fn = reward_fn_mapping[dataset]
    scores = [reward_fn(None, o, gt).score for o in outputs]
    return sum(scores) / len(scores) if scores else 0.0


def load_samples(files):
    """Load all JSONL lines into memory (fast on SSD)."""
    for file in files:
        with open(file, "r") as f:
            for line in f:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Filter a JSONL dataset by correctness using multiprocessing."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more .jsonl files produced by sampling"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save filtered samples"
    )
    parser.add_argument(
        "--lower_bound",
        type=float,
        default=0.0,
        help="Lower bound for correctness"
    )
    parser.add_argument(
        "--upper_bound",
        type=float,
        default=1.0,
        help="Upper bound for correctness"
    )
    parser.add_argument(
        "--hist_file_name",
        type=str,
        default="hist.png",
        help="Name of the histogram file"
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help="Give a dataset name to push this data to the hub."
    )
    parser.add_argument(
        "--code_api_url",
        default=None,
        type=str,
        help="Give a code api url to use for code verifier."
    )
    parser.add_argument(
        "--code_max_execution_time",
        default=1.0,
        type=float,
        help="Give a max execution time for code verifier."
    )
    parser.add_argument(
        "--llm_judge_model",
        default=None,
        type=str,
        help="Give a llm judge model to use for llm verifier."
    )
    parser.add_argument(
        "--llm_judge_max_tokens",
        default=2048,
        type=int,
        help="Give a max tokens for llm judge."
    )
    parser.add_argument(
        "--llm_judge_temperature",
        default=1.0,
        type=float,
        help="Give a temperature for llm judge."
    )
    parser.add_argument(
        "--llm_judge_timeout",
        default=60,
        type=int,
        help="Give a timeout for llm judge."
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Give a seed for llm judge."
    )
    args = parser.parse_args()
    if args.lower_bound == 0 and args.upper_bound == 1:
        print("Upper bound is 1 and lower bound is 0. No filtering will be done, is this intended?")

    reward_fn_mapping = build_all_verifiers(args)

    # currying the avg_correctness function
    avg_correctness = partial(_avg_correctness, reward_fn_mapping=reward_fn_mapping)

    # Prefer 'spawn' for better safety on macOS / Jupyter
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    samples = list(load_samples(args.files))

    # Multiprocess pool
    workers = min(cpu_count(), 32)  # Cap if you don’t want 128 cores
    chunk_size = 1  # Tune for workload size

    with Pool(processes=workers) as pool:
        avg_scores = list(
            tqdm(
                pool.imap(avg_correctness, samples, chunksize=chunk_size),
                total=len(samples),
                desc="Scoring"
            )
        )

    # Simple diagnostic plot
    plt.hist(avg_scores, bins=100)
    plt.xlabel("Average correctness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.hist_file_name)

    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    # Filter out everything outside of this range
    filtered_samples = [
        sample for sample, score in zip(samples, avg_scores)
        if lower_bound <= score <= upper_bound
    ]
    print(
        f"Filtered {len(samples) - len(filtered_samples)} samples out of {len(samples)}"
    )

    # Save filtered samples
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample) + "\n")

    if args.push_to_hub is not None:
        dataset = load_dataset(args.dataset, split=args.split)
        dataset.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
