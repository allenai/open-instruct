#!/usr/bin/env python3
"""
Filter an existing jsonl dataset by correctness, using multiple processes.
requires reward functions setup.
we use multiprocessing to make things actually fast.

to run:
python scripts/data/rlvr/filter_existing_dataset_correctness.py \
  --files data/*.jsonl --output_file filtered.jsonl

If you have code data, you might have to launch code server too before running:
bash configs/beaker_configs/code_api_setup.sh





python mason.py \
  --cluster ai2/augusta-google-1 \
  --image nathanl/open_instruct_auto --pure_docker_mode \
  --workspace ai2/oe-adapt-code \
  --description "filtering open code reasoning stdio full" \
  --priority high \
  --preemptible \
  --gpus 1 \
  --num_nodes 1 \
  --budget ai2/oe-adapt \
  --max_retries 0 \
  -- bash configs/beaker_configs/code_api_setup.sh && python scripts/data/rlvr/filter_existing_dataset_correctness.py \
  --dataset \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-0 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-10000 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-20000 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-30000 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-40000 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-50000 \
  saurabh5/saurabh5-rlvr_acecoder_filtered-offline-results-full-chunk-60000 \
--code_api_url http://localhost:1234/test_program \
--push_to_hub saurabh5/rlvr_acecoder_ot_diff_filtered \
--upper_bound 0.8


"""
import argparse
import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from datasets import Dataset, load_dataset

from open_instruct.ground_truth_utils import build_all_verifiers


def _calculate_metrics(sample, reward_fn_mapping):
    """
    Compute avg correctness and solvability for one sample.
    """
    dataset = sample["dataset"][0] if isinstance(sample["dataset"], list) else sample["dataset"]
    gt = sample["ground_truth"][0] if isinstance(sample["ground_truth"], list) else sample["ground_truth"]
    outputs = sample["output"]

    reward_fn = reward_fn_mapping[dataset]
    if not outputs:
        return 0.0, 0

    scores = [reward_fn(None, o, gt).score for o in outputs]
    avg_correctness = sum(scores) / len(scores) if scores else 0.0
    is_solvable = 1 if any(score == 1.0 for score in scores) else 0
    return avg_correctness, is_solvable


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
        default=None,
        help="One or more .jsonl files produced by sampling. Mutually exclusive with --dataset."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="One or more HF dataset names (e.g. squad). Mutually exclusive with --files."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to load from HF dataset"
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
        "--violin_plot_file_name",
        type=str,
        default="difficulty_score_violin.png",
        help="Name of the violin plot file for score vs difficulty."
    )
    parser.add_argument(
        "--solvable_plot_file_name",
        type=str,
        default="difficulty_solvable_bar.png",
        help="Name of the bar plot file for solvability rate vs difficulty."
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

    if not args.files and not args.dataset:
        parser.error("Either --files or --dataset must be specified.")
    if args.files and args.dataset:
        parser.error("Cannot specify both --files and --dataset.")

    if args.code_api_url is None:
        api_url_from_env = os.environ.get("CODE_API_URL")
        if api_url_from_env:
            api_url_from_env = api_url_from_env.strip()
        if not api_url_from_env:
            raise ValueError("CODE_API_URL environment variable not set and --code_api_url not provided.")
        args.code_api_url = f"{api_url_from_env}/test_program"

    if args.lower_bound == 0 and args.upper_bound == 1:
        print("Upper bound is 1 and lower bound is 0. No filtering will be done, is this intended?")

    reward_fn_mapping = build_all_verifiers(args)

    # currying the function
    calculate_metrics = partial(_calculate_metrics, reward_fn_mapping=reward_fn_mapping)

    # Prefer 'spawn' for better safety on macOS / Jupyter
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    if args.dataset:
        print(f"Loading and concatenating datasets from the Hub: {args.dataset}")
        samples = []
        for dataset_name in args.dataset:
            samples.extend(load_dataset(dataset_name, split=args.split))
    else:
        print(f"Loading samples from local files: {args.files}")
        samples = list(load_samples(args.files))

    # Multiprocess pool
    workers = min(cpu_count(), 32)  # Cap if you don't want 128 cores
    chunk_size = 1  # Tune for workload size

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_metrics, samples, chunksize=chunk_size),
                total=len(samples),
                desc="Scoring"
            )
        )
    avg_scores, solvable_flags = zip(*results)

    # Calculate correlation with difficulty
    difficulties = [sample.get("difficulty") for sample in samples]
    score_difficulty_pairs = [
        (score, difficulty / 10)
        for score, difficulty in zip(avg_scores, difficulties)
        if difficulty is not None
    ]

    if score_difficulty_pairs:
        if len(score_difficulty_pairs) > 1:
            # Unzip the pairs
            corr_scores, corr_difficulties = zip(*score_difficulty_pairs)
            # Calculate correlation
            correlation = np.corrcoef(corr_scores, corr_difficulties)[0, 1]
            print(f"Correlation between score and difficulty: {correlation:.4f}")

            # Create violin plot of score distribution by difficulty
            original_difficulties = [d for d in difficulties if d is not None]
            plt.figure(figsize=(12, 7))
            sns.violinplot(x=original_difficulties, y=list(corr_scores))
            plt.xlabel("Difficulty")
            plt.ylabel("Average Correctness Score")
            plt.title("Distribution of Correctness Scores by Difficulty")
            plt.tight_layout()
            plt.savefig(args.violin_plot_file_name)
            print(f"Saved score vs difficulty violin plot to {args.violin_plot_file_name}")
        else:
            print("Not enough data points with 'difficulty' (need at least 2) to calculate correlation or create distribution plot.")
    else:
        print("No samples with 'difficulty' field found. Cannot calculate correlation or create distribution plot.")

    # Calculate correlation and plot for solvable vs difficulty
    solvable_difficulty_pairs = [
        (solvable, difficulty)
        for solvable, difficulty in zip(solvable_flags, difficulties)
        if difficulty is not None
    ]

    if len(solvable_difficulty_pairs) > 1:
        corr_solvable, corr_difficulties_s = zip(*solvable_difficulty_pairs)
        correlation_solvable = np.corrcoef(corr_solvable, corr_difficulties_s)[0, 1]
        print(f"Correlation between solvable and difficulty: {correlation_solvable:.4f}")

        # Create bar plot of solvability rate by difficulty
        solvability_by_diff = defaultdict(list)
        for diff, solv in zip(corr_difficulties_s, corr_solvable):
            solvability_by_diff[diff].append(solv)

        difficulty_levels = sorted(solvability_by_diff.keys())
        solvability_rates = [np.mean(solvability_by_diff[d]) for d in difficulty_levels]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=difficulty_levels, y=solvability_rates, palette="viridis")
        plt.xlabel("Difficulty")
        plt.ylabel("Solvability Rate (Proportion of `any` perfect scores)")
        plt.title("Solvability Rate by Difficulty")
        plt.tight_layout()
        plt.savefig(args.solvable_plot_file_name)
        print(f"Saved solvability rate vs difficulty bar plot to {args.solvable_plot_file_name}")

    # Simple diagnostic plot
    plt.figure()
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
        if not filtered_samples:
            print("Warning: No samples were left after filtering. Nothing to push to Hub.")
        else:
            print(f"Pushing {len(filtered_samples)} filtered samples to {args.push_to_hub} on the Hub.")
            # Create a HF dataset from the filtered list of dicts
            filtered_ds = Dataset.from_list(filtered_samples)
            filtered_ds.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
