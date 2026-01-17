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

You might have to explicitly install nginx first: sudo apt-get update && apt-get install -y --no-install-recommends nginx
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import VerifierFunction, build_all_verifiers

logger = logger_utils.setup_logger(__name__)


def _avg_correctness(
    sample: dict, reward_fn_mapping: dict[str, VerifierFunction], judge_override: str | None = None
) -> tuple[float, int]:
    """
    Compute the mean correctness for one sample (called in worker).
    Args:
        sample: The sample to compute the correctness for. Should have "dataset", "ground_truth", and "output" keys. Output should be a list of strings (list of completions for the sample).
        reward_fn_mapping: The reward function mapping. Should be a dictionary of verifier names to verifier functions objects.
        judge_override: If specified, use this judge/verifier for all samples instead of the dataset-provided one.
    Returns:
        The average score of outputs as judged by the verifier function. If there are no outputs, return 0.0.
        The number of outputs.
    """
    dataset = sample["dataset"][0] if isinstance(sample["dataset"], list) else sample["dataset"]
    gt = sample["ground_truth"][0] if isinstance(sample["ground_truth"], list) else sample["ground_truth"]
    outputs = sample["output"] if "output" in sample else sample["outputs"]

    query = None
    if "messages" in sample:
        query = "\n".join(f"{msg['role']}: {msg['content']}" for msg in sample["messages"])

    key = judge_override if judge_override is not None else dataset
    reward_fn = reward_fn_mapping[key]
    scores = [reward_fn(None, o, gt, query).score for o in outputs]
    return sum(scores) / len(scores) if scores else 0.0, len(scores)


def load_samples(files):
    """Load all JSONL lines into memory (fast on SSD)."""
    for file in files:
        with open(file) as f:
            for line in f:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Filter a JSONL dataset by correctness using multiprocessing.")
    parser.add_argument("--files", nargs="+", required=True, help="One or more .jsonl files produced by sampling")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save filtered samples")
    parser.add_argument("--lower_bound", type=float, default=0.0, help="Lower bound for correctness")
    parser.add_argument("--upper_bound", type=float, default=1.0, help="Upper bound for correctness")
    parser.add_argument("--hist_file_name", type=str, default="hist.png", help="Name of the histogram file")
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Give a dataset name to push this data to the hub."
    )
    parser.add_argument(
        "--code_api_url",
        default=os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program",
        type=str,
        help="Give a code api url to use for code verifier.",
    )
    parser.add_argument(
        "--code_max_execution_time", default=1.0, type=float, help="Give a max execution time for code verifier."
    )
    parser.add_argument(
        "--code_pass_rate_reward_threshold",
        default=1.0,
        type=float,
        help="Threshold for pass rate; below this the code verifier returns 0.0.",
    )
    parser.add_argument(
        "--code_apply_perf_penalty",
        action="store_true",
        help="If set, apply a runtime-based performance penalty to passing code tests.",
    )
    parser.add_argument(
        "--llm_judge_model", default=None, type=str, help="Give a llm judge model to use for llm verifier."
    )
    parser.add_argument("--llm_judge_max_tokens", default=2048, type=int, help="Give a max tokens for llm judge.")
    parser.add_argument(
        "--llm_judge_max_context_length", default=128_000, type=int, help="Max context length for the LLM judge model."
    )
    parser.add_argument("--llm_judge_temperature", default=1.0, type=float, help="Give a temperature for llm judge.")
    parser.add_argument("--llm_judge_timeout", default=60, type=int, help="Give a timeout for llm judge.")
    parser.add_argument("--seed", default=42, type=int, help="Give a seed for llm judge.")
    parser.add_argument(
        "--max_length_verifier_max_length",
        default=32768,
        type=int,
        help="Max length used by max-length style verifiers.",
    )
    parser.add_argument(
        "--remap_verifier", default=None, type=str, help="Optional mapping old=new to remap a verifier name."
    )
    parser.add_argument("--split", type=str, default="train", help="Split to use on upload")
    parser.add_argument(
        "--annotate_original_dataset",
        type=str,
        default=None,
        help="If set, annotate the original dataset with the passrates, and save to this path.",
    )
    parser.add_argument(
        "--judge_override",
        type=str,
        default=None,
        help=(
            "If set, use this judge/verifier for all samples instead of the dataset-provided one. "
            "Accepts keys from build_all_verifiers(), e.g. 'math', 'string_f1', 'code', or 'general-quality'. "
            "For LLM judges, you may also pass just the judge type like 'quality' which will map to 'general-quality'."
        ),
    )
    args = parser.parse_args()
    if args.lower_bound == 0 and args.upper_bound == 1:
        logger.warning("Upper bound is 1 and lower bound is 0. No filtering will be done, is this intended?")

    reward_fn_mapping = build_all_verifiers(args)

    # Resolve judge override if provided
    override_key = None
    if args.judge_override is not None:
        candidate = args.judge_override.lower()
        if candidate not in reward_fn_mapping and f"general-{candidate}" in reward_fn_mapping:
            candidate = f"general-{candidate}"
        if candidate not in reward_fn_mapping:
            raise ValueError(
                f"Judge override '{args.judge_override}' not found in available verifiers. "
                f"Try one of: {', '.join(sorted(reward_fn_mapping.keys()))}"
            )
        override_key = candidate
        logger.info(f"Using judge override: {override_key}")

    # currying the avg_correctness function
    avg_correctness = partial(_avg_correctness, reward_fn_mapping=reward_fn_mapping, judge_override=override_key)

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
        results = list(
            tqdm(pool.imap(avg_correctness, samples, chunksize=chunk_size), total=len(samples), desc="Scoring")
        )
    # results is a list of tuples: (avg_score, num_rollouts)
    avg_scores = [score for score, _ in results]
    num_rollouts = [n for _, n in results]

    # Simple diagnostic plot
    plt.hist(avg_scores, bins=100)
    plt.xlabel("Average correctness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.hist_file_name)

    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    # Filter out everything outside of this range
    filtered_samples = [sample for sample, score in zip(samples, avg_scores) if lower_bound <= score <= upper_bound]
    logger.info(f"Filtered {len(samples) - len(filtered_samples)} samples out of {len(samples)}")

    # Save filtered samples
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample) + "\n")

    # Annotate the original dataset with the passrates if requested, and save.
    if args.annotate_original_dataset is not None:
        for sample, num_r, score in zip(samples, num_rollouts, avg_scores):
            sample["total_rollouts"] = num_r
            sample["total_correct_rollouts"] = score * num_r
            sample["passrate"] = score
        with open(args.annotate_original_dataset, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    if args.push_to_hub is not None:
        dataset = Dataset.from_list(filtered_samples)
        dataset.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
