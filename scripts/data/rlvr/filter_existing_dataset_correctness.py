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
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import matplotlib.pyplot as plt
from open_instruct.ground_truth_utils import build_all_verifiers


def _avg_correctness(sample):
    """
    Compute the mean correctness for one sample (called in worker).
    """
    dataset = sample["dataset"][0]
    gt = sample["ground_truth"][0]
    outputs = sample["output"]

    reward_fn = REWARD_FN_MAPPING[dataset]
    scores = [reward_fn(None, o, gt) for o in outputs]
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
        required=True,
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
    args = parser.parse_args()

    reward_fn_mapping = build_all_verifiers(args)

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
                pool.imap(_avg_correctness, samples, chunksize=chunk_size),
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
        if lower_bound < score < upper_bound
    ]
    print(
        f"Filtered {len(samples) - len(filtered_samples)} samples out of {len(samples)}"
    )

    # Save filtered samples
    with open(args.output_file, "w") as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
