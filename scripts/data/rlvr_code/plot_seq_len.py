"""
Sequence Length Distribution Plotter for HuggingFace Datasets

This script analyzes the sequence length distribution of text data in HuggingFace datasets.
It tokenizes a specified column using a pre-trained tokenizer and generates a histogram
showing the distribution of sequence lengths, along with percentile markers and statistics.

Features:
- Supports both streaming and non-streaming dataset loading
- Configurable tokenizer, column selection, and plotting parameters
- Adds percentile lines (10%, 20%, ..., 90%) to the histogram
- Displays min/max/average/median statistics
- Saves plots to file and optionally displays them
- Handles large datasets efficiently with batched processing

Usage:
    python plot_seq_len.py --dataset_name <dataset> --split <split> --column_name <column>

Required Arguments:
    --dataset_name: HuggingFace dataset name (e.g., "allenai/c4")
    --split: Dataset split to analyze (e.g., "train", "validation")
    --column_name: Column containing text data to analyze (default: "output")

Optional Arguments:
    --batch_size: Batch size for processing (default: 1000)
    --num_bins: Number of histogram bins (default: 50)
    --plot_filename: Output filename for the plot
    --streaming: Enable streaming mode for large datasets
    --max_samples_streaming: Limit samples in streaming mode (0 for all)
    --show_plot: Display the plot after saving

Examples:
    # Basic usage
    python plot_seq_len.py --dataset_name "allenai/c4" --split "train" --column_name "text"

    # Streaming mode for large datasets
    python plot_seq_len.py --dataset_name "allenai/c4" --split "train" --column_name "text" --streaming --max_samples_streaming 10000

    # Custom plot settings
    python plot_seq_len.py --dataset_name "my/dataset" --split "validation" --column_name "output" --num_bins 100 --show_plot
"""

import argparse
import sys

import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-SFT"


# --- Helper Function ---
def get_sequence_lengths(batch, tokenizer, column_name):
    """Tokenizes the specified column and returns sequence lengths."""
    if column_name not in batch:
        if not hasattr(get_sequence_lengths, "warned"):
            logger.warning(f"Column '{column_name}' not found in at least one batch. Skipping.")
            get_sequence_lengths.warned = True
        return {"seq_len": []}

    outputs = [str(item) if item is not None else "" for item in batch[column_name]]
    try:
        tokenized_outputs = tokenizer(outputs, truncation=False, padding=False)
        return {"seq_len": [len(ids) for ids in tokenized_outputs["input_ids"]]}
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        return {"seq_len": [0] * len(outputs)}


# --- Main Function ---
def main(args):
    """Loads dataset, calculates lengths, and plots distribution."""
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer {MODEL_NAME}: {e}")
        sys.exit(1)

    logger.info(f"Loading dataset: {args.dataset_name}, split: {args.split}")
    dataset = None
    try:
        dataset = datasets.load_dataset(
            args.dataset_name, split=args.split, streaming=args.streaming, num_proc=max_num_processes()
        )
        if args.streaming:
            logger.info("Processing in streaming mode.")
            if args.max_samples_streaming > 0:
                logger.info(f"Taking first {args.max_samples_streaming} samples.")
                dataset = dataset.take(args.max_samples_streaming)

    except FileNotFoundError:
        logger.error(f"Dataset '{args.dataset_name}' not found.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid split '{args.split}' or dataset config error for '{args.dataset_name}': {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_name}' split '{args.split}': {e}")
        sys.exit(1)

    if not args.streaming and dataset:
        if args.column_name not in dataset.column_names:
            logger.error(
                f"Column '{args.column_name}' not found in dataset '{args.dataset_name}' split '{args.split}'."
            )
            logger.info(f"Available columns: {dataset.column_names}")
            sys.exit(1)

    logger.info(f"Calculating sequence lengths for column '{args.column_name}'...")
    sequence_lengths = []
    try:
        if hasattr(get_sequence_lengths, "warned"):
            delattr(get_sequence_lengths, "warned")

        mapped_dataset = dataset.map(
            get_sequence_lengths,
            fn_kwargs={"tokenizer": tokenizer, "column_name": args.column_name},
            batched=True,
            batch_size=args.batch_size,
        )

        logger.info("Collecting sequence lengths...")
        if args.streaming:
            temp_lengths = []
            # Correctly iterate over batches in streaming mode
            for batch in tqdm(mapped_dataset, desc="Processing batches"):
                if "seq_len" in batch:
                    # Extend with lengths from the current batch
                    batch_lengths = batch["seq_len"]
                    # Ensure batch_lengths is a list, even if map returns single value for single input
                    if isinstance(batch_lengths, list):
                        temp_lengths.extend(batch_lengths)
                    else:  # Handle case where maybe it's not a list (less common for batched=True)
                        temp_lengths.append(batch_lengths)
                else:
                    # This might happen if get_sequence_lengths returned empty due to missing column
                    logger.debug("'seq_len' key missing in a batch.")

            sequence_lengths = temp_lengths
        else:
            sequence_lengths = mapped_dataset["seq_len"]

    except KeyError:
        logger.error(f"Column '{args.column_name}' caused a KeyError during processing.")
        try:
            if not args.streaming and dataset:
                logger.info(f"Available columns: {dataset.column_names}")
            else:
                logger.info("Cannot list columns for streaming dataset easily after error.")
        except Exception as e_inner:
            logger.warning(f"Could not retrieve column names: {e_inner}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error calculating sequence lengths: {e}")
        import traceback

        traceback.print_exc()  # Print stack trace for debugging
        sys.exit(1)

    if not sequence_lengths:
        logger.warning("No sequence lengths were calculated. Check dataset, column name, and potential errors.")
        sys.exit(1)

    logger.info(f"Successfully calculated {len(sequence_lengths)} sequence lengths.")
    min_len = np.min(sequence_lengths)
    max_len = np.max(sequence_lengths)
    avg_len = np.mean(sequence_lengths)
    median_len = np.median(sequence_lengths)
    logger.info(
        f"Min length: {min_len}, Max length: {max_len}, Avg length: {avg_len:.2f}, Median length: {median_len}"
    )

    logger.info("Plotting histogram...")
    plt.figure(figsize=(12, 7))
    num_bins = min(args.num_bins, int(max_len / 10) if max_len > 10 else args.num_bins)  # Adjust bin calculation
    if num_bins <= 0:
        num_bins = 10
    plt.hist(sequence_lengths, bins=num_bins, color="skyblue", edgecolor="black")

    # --- Add Percentile Lines ---
    percentiles = np.arange(10, 100, 10)  # 10, 20, ..., 90
    percentile_values = np.percentile(sequence_lengths, percentiles)
    logger.info(f"Percentile values (10th-90th): {dict(zip(percentiles, np.round(percentile_values, 2)))}")

    # Determine a good y-position for labels (e.g., 95% of max y-axis height)
    y_max = plt.gca().get_ylim()[1]
    label_y_pos = y_max * 0.95

    for p, val in zip(percentiles, percentile_values):
        plt.axvline(val, color="red", linestyle="dashed", linewidth=1)
        # Add text label slightly above the line, adjusting x position slightly
        plt.text(val * 1.01, label_y_pos, f"{p}%", color="red", verticalalignment="top", fontsize=8)
    # ---------------------------

    plt.title(
        f'Distribution of Sequence Lengths for "{args.column_name}" column\nDataset: {args.dataset_name} ({args.split} split) - {len(sequence_lengths)} samples'
    )
    plt.xlabel("Sequence Length (Tokens)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    stats_text = f"Min: {min_len}\nMax: {max_len}\nAvg: {avg_len:.2f}\nMedian: {median_len}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plot_filename = (
        args.plot_filename or f"{args.dataset_name.replace('/', '_')}_{args.split}_{args.column_name}_seq_len_dist.png"
    )
    try:
        plt.savefig(plot_filename)
        logger.info(f"Histogram saved to {plot_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_filename}: {e}")

    if args.show_plot:
        logger.info("Displaying plot...")
        plt.show()


# --- Argument Parsing and Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot sequence length distribution for a Hugging Face dataset column."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help='Name of the Hugging Face dataset (e.g., "allenai/c4").'
    )
    parser.add_argument(
        "--split", type=str, default="train", help='Dataset split to use (e.g., "train", "validation").'
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="output",
        help='Name of the column containing text data (default: "output").',
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for mapping function (default: 1000)."
    )
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins for the histogram (default: 50).")
    parser.add_argument(
        "--plot_filename",
        type=str,
        default=None,
        help="Filename to save the plot. Defaults to DATANAME_SPLIT_COL_seq_len_dist.png",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Load dataset in streaming mode (useful for large datasets)."
    )
    parser.add_argument(
        "--max_samples_streaming",
        type=int,
        default=0,
        help="Max samples to process in streaming mode (0 for all, default: 0). Use with caution for very large datasets.",
    )
    parser.add_argument("--show_plot", action="store_true", help="Display the plot after saving.")

    args = parser.parse_args()

    if args.streaming and args.max_samples_streaming < 0:
        logger.error("--max_samples_streaming cannot be negative.")
        sys.exit(1)
    elif not args.streaming and args.max_samples_streaming != 0:
        logger.warning("--max_samples_streaming is ignored when --streaming is not used.")

    main(args)
