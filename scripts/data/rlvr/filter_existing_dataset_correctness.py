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
import os
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

import matplotlib.pyplot as plt
from tqdm import tqdm

from open_instruct.ground_truth_utils import build_all_verifiers, LMJudgeVerifier


def _avg_correctness(sample, reward_fn_mapping):
    """
    Compute the mean correctness for one sample (called in worker).
    """
    dataset = sample["dataset"][0] if type(sample["dataset"]) == list else sample["dataset"]
    gt = sample["ground_truth"][0] if type(sample["ground_truth"]) == list else sample["ground_truth"]
    outputs = sample["outputs"]

    reward_fn = reward_fn_mapping[dataset]
    if isinstance(reward_fn, LMJudgeVerifier):
        scores = [reward_fn(None, o, gt, sample["prompt"]).score if o else 0.0 for o in outputs]
    else:
        scores = [reward_fn(None, o, gt).score if o else 0.0 for o in outputs]

    overall_score = sum(scores) / len(scores) if scores else 0.0
    return overall_score, scores


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
    # -- llm judge
    parser.add_argument(
        "--llm_judge_model",
        default="hosted_vllm/Qwen/Qwen3-32B",
        type=str,
        help="the model to use for the llm judge"
    )
    parser.add_argument(
        "--llm_judge_max_tokens",
        default=2048,
        type=int,
        help="the max tokens to use for the llm judge"
    )
    parser.add_argument(
        "--llm_judge_max_context_length",
        default=8192,
        type=int,
        help="the max context length to use for the llm judge"
    )
    parser.add_argument(
        "--llm_judge_temperature",
        default=1.0,
        type=float,
        help="the temperature to use for the llm judge"
    )
    parser.add_argument(
        "--llm_judge_timeout",
        default=600,
        type=int,
        help="the timeout to use for the llm judge"
    )

    # -- code verifier
    parser.add_argument(
        "--code_api_url",
        default=os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program",
        type=str,
        help="the api url to use for the code verifier"
    )
    parser.add_argument(
        "--code_max_execution_time",
        default=1.0,
        type=float,
        help="the max execution time to use for the code verifier"
    )
    parser.add_argument(
        "--code_pass_rate_reward_threshold",
        default=0.0,
        type=float,
        help="the pass rate reward threshold for the code verifier. If pass rate is less than this threshold, reward is 0.0, otherwise reward is pass rate"
    )
    parser.add_argument(
        "--code_apply_perf_penalty",
        default=False,
        type=bool,
        help="whether to apply a performance penalty to the code verifier"
    )

    # -- max length verifier
    parser.add_argument(
        "--max_length_verifier_max_length",
        default=32768,
        type=int,
        help="the max length to use for the max length verifier"
    )

    # -- non stop penalty
    parser.add_argument(
        "--non_stop_penalty",
        default=False,
        type=bool,
        help="whether to penalize responses which did not finish generation"
    )
    parser.add_argument(
        "--non_stop_penalty_value",
        default=0.0,
        type=float,
        help="the reward value for responses which did not finish generation"
    )
    parser.add_argument(
        "--remap_verifier",
        default=None,
        type=str,
        help="Remap verifier like string_f1=general-quality_ref. Currently can only remap once."
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
        scored_outputs = list(
            tqdm(
                pool.imap(avg_correctness, samples, chunksize=chunk_size),
                total=len(samples),
                desc="Scoring"
            )
        )
        avg_scores, scores = zip(*scored_outputs)

    # Simple diagnostic plot
    plt.hist(avg_scores, bins=100)
    plt.xlabel("Average correctness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.hist_file_name)

    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    # Filter out everything outside of this range
    filtered_samples = []
    for sample, avg_score, full_scores in zip(samples, avg_scores, scores):
        if lower_bound <= avg_score <= upper_bound:
            sample["scores"] = full_scores
            filtered_samples.append(sample)

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
