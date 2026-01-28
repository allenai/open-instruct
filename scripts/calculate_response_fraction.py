#!/usr/bin/env python
"""
Calculate response fraction for DPO datasets.

This script calculates the fraction of tokens that are response tokens (labels != -100)
vs total tokens in the dataset. This is useful for comparing throughput metrics between
different training implementations.

Usage:
    uv run python scripts/calculate_response_fraction.py --script scripts/train/olmo3/7b_instruct_dpo_olmocore.sh
    uv run python scripts/calculate_response_fraction.py --mixer_list allenai/dataset1 1.0 allenai/dataset2 0.5 \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B
"""

import argparse
import re
import shlex
import sys

import numpy as np
from tqdm import tqdm

from open_instruct import logger_utils
from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    CHOSEN_LABELS_KEY,
    REJECTED_INPUT_IDS_KEY,
    REJECTED_LABELS_KEY,
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
)

logger = logger_utils.setup_logger(__name__)


def parse_script_for_args(script_path: str) -> dict:
    """Parse a training script to extract relevant arguments."""
    with open(script_path) as f:
        content = f.read()

    content = content.replace("\\\n", " ")
    lines = content.split("\n")
    command_lines = [line for line in lines if not line.strip().startswith("#") and line.strip()]
    full_command = " ".join(command_lines)

    args = {}
    mixer_match = re.search(r"--mixer_list\s+(.+?)(?=\s+--\w+|\s*$)", full_command, re.DOTALL)
    if mixer_match:
        mixer_str = mixer_match.group(1).strip()
        mixer_parts = shlex.split(mixer_str)
        args["mixer_list"] = mixer_parts

    max_seq_match = re.search(r"--max_seq_length\s+(\d+)", full_command)
    if max_seq_match:
        args["max_seq_length"] = int(max_seq_match.group(1))

    tokenizer_match = re.search(r"--tokenizer_name_or_path\s+(\S+)", full_command)
    if tokenizer_match:
        args["tokenizer_name_or_path"] = tokenizer_match.group(1)

    model_match = re.search(r"--model_name_or_path\s+(\S+)", full_command)
    if model_match:
        args["model_name_or_path"] = model_match.group(1)

    chat_template_match = re.search(r"--chat_template_name\s+(\S+)", full_command)
    if chat_template_match:
        args["chat_template_name"] = chat_template_match.group(1)

    return args


def calculate_response_fraction(
    mixer_list: list[str],
    tc: TokenizerConfig,
    max_seq_length: int,
    max_samples: int | None = None,
) -> dict:
    """Calculate response fraction statistics for the dataset."""
    transform_fn_args = [{"max_seq_length": max_seq_length}, {}]

    logger.info("Loading dataset...")
    dataset = get_cached_dataset_tulu(
        dataset_mixer_list=mixer_list,
        dataset_mixer_list_splits=["train"],
        tc=tc,
        dataset_transform_fn=["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"],
        transform_fn_args=transform_fn_args,
        target_columns=TOKENIZED_PREFERENCE_DATASET_KEYS,
        dataset_skip_cache=False,
    )

    if max_samples is not None and len(dataset) > max_samples:
        logger.info(f"Sampling {max_samples} from {len(dataset)} total samples")
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    logger.info(f"Calculating response fraction for {len(dataset)} samples...")

    response_fractions = []
    chosen_response_lengths = []
    rejected_response_lengths = []
    chosen_total_lengths = []
    rejected_total_lengths = []

    for sample in tqdm(dataset, desc="Processing samples"):
        chosen_input_ids = sample[CHOSEN_INPUT_IDS_KEY]
        chosen_labels = sample[CHOSEN_LABELS_KEY]
        rejected_input_ids = sample[REJECTED_INPUT_IDS_KEY]
        rejected_labels = sample[REJECTED_LABELS_KEY]

        chosen_response_tokens = sum(1 for label in chosen_labels if label != -100)
        rejected_response_tokens = sum(1 for label in rejected_labels if label != -100)
        chosen_total_tokens = len(chosen_input_ids)
        rejected_total_tokens = len(rejected_input_ids)

        total_response_tokens = chosen_response_tokens + rejected_response_tokens
        total_tokens = chosen_total_tokens + rejected_total_tokens

        if total_tokens > 0:
            response_fraction = total_response_tokens / total_tokens
            response_fractions.append(response_fraction)

        chosen_response_lengths.append(chosen_response_tokens)
        rejected_response_lengths.append(rejected_response_tokens)
        chosen_total_lengths.append(chosen_total_tokens)
        rejected_total_lengths.append(rejected_total_tokens)

    response_fractions = np.array(response_fractions)
    chosen_response_lengths = np.array(chosen_response_lengths)
    rejected_response_lengths = np.array(rejected_response_lengths)
    chosen_total_lengths = np.array(chosen_total_lengths)
    rejected_total_lengths = np.array(rejected_total_lengths)

    return {
        "num_samples": len(dataset),
        "response_fraction": {
            "mean": float(np.mean(response_fractions)),
            "std": float(np.std(response_fractions)),
            "min": float(np.min(response_fractions)),
            "max": float(np.max(response_fractions)),
            "median": float(np.median(response_fractions)),
            "p5": float(np.percentile(response_fractions, 5)),
            "p95": float(np.percentile(response_fractions, 95)),
        },
        "chosen_response_length": {
            "mean": float(np.mean(chosen_response_lengths)),
            "std": float(np.std(chosen_response_lengths)),
            "min": float(np.min(chosen_response_lengths)),
            "max": float(np.max(chosen_response_lengths)),
        },
        "rejected_response_length": {
            "mean": float(np.mean(rejected_response_lengths)),
            "std": float(np.std(rejected_response_lengths)),
            "min": float(np.min(rejected_response_lengths)),
            "max": float(np.max(rejected_response_lengths)),
        },
        "chosen_total_length": {
            "mean": float(np.mean(chosen_total_lengths)),
            "std": float(np.std(chosen_total_lengths)),
            "min": float(np.min(chosen_total_lengths)),
            "max": float(np.max(chosen_total_lengths)),
        },
        "rejected_total_length": {
            "mean": float(np.mean(rejected_total_lengths)),
            "std": float(np.std(rejected_total_lengths)),
            "min": float(np.min(rejected_total_lengths)),
            "max": float(np.max(rejected_total_lengths)),
        },
    }


def print_conversion_examples(stats: dict, num_gpus: int | None = None) -> None:
    """Print example conversions between old and new throughput metrics."""
    response_fraction = stats["response_fraction"]["mean"]

    print("\n" + "=" * 70)
    print("METRIC CONVERSION GUIDE")
    print("=" * 70)
    print(f"\nResponse fraction for this dataset: {response_fraction:.4f} ({response_fraction * 100:.2f}%)")

    if num_gpus is not None:
        print(f"\nFor {num_gpus} GPUs:")
        print("\nTo convert OLD metric (perf/tokens_per_second_total) to NEW metric (throughput/device/TPS):")
        print("  throughput/device/TPS = perf/tokens_per_second_total / (num_gpus × response_fraction)")
        print(f"  throughput/device/TPS = perf/tokens_per_second_total / ({num_gpus} × {response_fraction:.4f})")
        print(f"  throughput/device/TPS = perf/tokens_per_second_total / {num_gpus * response_fraction:.4f}")

        print("\n  Example conversions:")
        for old_tps in [50, 100, 150, 200, 250]:
            new_tps = old_tps / (num_gpus * response_fraction)
            print(f"    perf/tokens_per_second_total = {old_tps:4d} -> throughput/device/TPS = {new_tps:.1f}")

        print("\nTo convert NEW metric to OLD metric:")
        print("  perf/tokens_per_second_total = throughput/device/TPS × num_gpus × response_fraction")
        print(f"  perf/tokens_per_second_total = throughput/device/TPS × {num_gpus} × {response_fraction:.4f}")

        print("\n  Example conversions:")
        for new_tps in [100, 200, 500, 1000, 2000]:
            old_tps = new_tps * num_gpus * response_fraction
            print(f"    throughput/device/TPS = {new_tps:4d} -> perf/tokens_per_second_total = {old_tps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Calculate response fraction for DPO datasets")
    parser.add_argument(
        "--script",
        type=str,
        help="Path to a training script to parse for dataset configuration",
    )
    parser.add_argument(
        "--mixer_list",
        type=str,
        nargs="+",
        help="Dataset mixer list (alternating dataset names and fractions)",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--chat_template_name",
        type=str,
        default="olmo123",
        help="Chat template name (default: olmo123)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Maximum sequence length (default: 16384)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process (default: 1000). Set to 0 for all samples.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for conversion examples",
    )
    args = parser.parse_args()

    if args.script:
        script_args = parse_script_for_args(args.script)
        logger.info(f"Parsed from script: {script_args}")

        mixer_list = args.mixer_list or script_args.get("mixer_list")
        tokenizer_name_or_path = args.tokenizer_name_or_path or script_args.get(
            "tokenizer_name_or_path", script_args.get("model_name_or_path")
        )
        max_seq_length = args.max_seq_length or script_args.get("max_seq_length", 16384)
        chat_template_name = args.chat_template_name or script_args.get("chat_template_name", "olmo123")
    else:
        mixer_list = args.mixer_list
        tokenizer_name_or_path = args.tokenizer_name_or_path
        max_seq_length = args.max_seq_length
        chat_template_name = args.chat_template_name

    if not mixer_list:
        parser.error("Either --script or --mixer_list must be provided")
    if not tokenizer_name_or_path:
        parser.error("Either --script or --tokenizer_name_or_path must be provided")

    logger.info(f"Mixer list: {mixer_list}")
    logger.info(f"Tokenizer: {tokenizer_name_or_path}")
    logger.info(f"Max seq length: {max_seq_length}")
    logger.info(f"Chat template: {chat_template_name}")

    tc = TokenizerConfig(
        tokenizer_name_or_path=tokenizer_name_or_path,
        chat_template_name=chat_template_name,
    )

    max_samples = args.max_samples if args.max_samples > 0 else None

    stats = calculate_response_fraction(
        mixer_list=mixer_list,
        tc=tc,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
    )

    print("\n" + "=" * 70)
    print("RESPONSE FRACTION STATISTICS")
    print("=" * 70)
    print(f"\nNumber of samples analyzed: {stats['num_samples']}")

    print("\nResponse Fraction (response_tokens / total_tokens):")
    rf = stats["response_fraction"]
    print(f"  Mean:   {rf['mean']:.4f} ({rf['mean'] * 100:.2f}%)")
    print(f"  Std:    {rf['std']:.4f}")
    print(f"  Min:    {rf['min']:.4f}")
    print(f"  Max:    {rf['max']:.4f}")
    print(f"  Median: {rf['median']:.4f}")
    print(f"  P5:     {rf['p5']:.4f}")
    print(f"  P95:    {rf['p95']:.4f}")

    print("\nChosen Response Length (tokens where labels != -100):")
    crl = stats["chosen_response_length"]
    print(f"  Mean: {crl['mean']:.1f}, Std: {crl['std']:.1f}, Min: {crl['min']:.0f}, Max: {crl['max']:.0f}")

    print("\nRejected Response Length (tokens where labels != -100):")
    rrl = stats["rejected_response_length"]
    print(f"  Mean: {rrl['mean']:.1f}, Std: {rrl['std']:.1f}, Min: {rrl['min']:.0f}, Max: {rrl['max']:.0f}")

    print("\nChosen Total Length:")
    ctl = stats["chosen_total_length"]
    print(f"  Mean: {ctl['mean']:.1f}, Std: {ctl['std']:.1f}, Min: {ctl['min']:.0f}, Max: {ctl['max']:.0f}")

    print("\nRejected Total Length:")
    rtl = stats["rejected_total_length"]
    print(f"  Mean: {rtl['mean']:.1f}, Std: {rtl['std']:.1f}, Min: {rtl['min']:.0f}, Max: {rtl['max']:.0f}")

    print_conversion_examples(stats, args.num_gpus)


if __name__ == "__main__":
    main()
