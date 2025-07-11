"""
Difficulty Distribution Plotter for HuggingFace Datasets

This script analyzes the difficulty distribution of data in HuggingFace datasets.
It reads an integer 'difficulty' column and generates a histogram
showing the distribution of these scores.

Features:
- Supports multiple datasets for comparison
- Supports both streaming and non-streaming dataset loading
- Configurable column selection and plotting parameters
- Displays min/max/average/median statistics for each dataset
- Saves plots to file and optionally displays them
- Handles large datasets efficiently with batched processing

Usage:
    python plot_difficulty.py --dataset_names <dataset1> <dataset2> --split <split>

Required Arguments:
    --dataset_names: One or more HuggingFace dataset names (e.g., "allenai/c4" "another/dataset")

Optional Arguments:
    --split: Dataset split to analyze (e.g., "train", "validation") (default: "train")
    --column_name: Column containing integer difficulty scores (default: "difficulty")
    --batch_size: Batch size for processing (default: 1000)
    --num_bins: Number of histogram bins (default: 50)
    --plot_filename: Output filename for the plot
    --streaming: Enable streaming mode for large datasets
    --max_samples_streaming: Limit samples in streaming mode (0 for all)
    --show_plot: Display the plot after saving

Examples:
    # Basic usage with one dataset
    python plot_difficulty.py --dataset_names "my/difficulty_dataset" --split "train"

    # Comparing two datasets
    python plot_difficulty.py --dataset_names "dataset/v1" "dataset/v2" --split "validation" --show_plot

    # Streaming mode for a large dataset
    python plot_difficulty.py --dataset_names "large/dataset" --streaming --max_samples_streaming 10000
"""

import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def get_difficulty_scores(batch, column_name):
    """Extracts integer scores from the specified column."""
    if column_name not in batch:
        if not hasattr(get_difficulty_scores, "warned"):
             logging.warning(f"Column '{column_name}' not found in at least one batch. Skipping.")
             get_difficulty_scores.warned = True
        return {"difficulty": []}

    try:
        # Assuming the values are integers. Handle potential errors.
        scores = [int(item) for item in batch[column_name] if item is not None]
        return {"difficulty": scores}
    except (ValueError, TypeError) as e:
        logging.error(f"Error converting values in '{column_name}' to int: {e}. Skipping batch.")
        return {"difficulty": []}


# --- Main Function ---
def main(args):
    """Loads datasets, calculates difficulty scores, and plots distribution."""

    all_difficulty_scores = {}

    for dataset_name in args.dataset_names:
        logging.info(f"Loading dataset: {dataset_name}, split: {args.split}")
        dataset = None
        try:
            dataset = datasets.load_dataset(dataset_name, split=args.split, streaming=args.streaming)
            if args.streaming:
                logging.info("Processing in streaming mode.")
                if args.max_samples_streaming > 0:
                    logging.info(f"Taking first {args.max_samples_streaming} samples.")
                    dataset = dataset.take(args.max_samples_streaming)

        except FileNotFoundError:
            logging.error(f"Dataset '{dataset_name}' not found. Skipping.")
            continue
        except ValueError as e:
             logging.error(f"Invalid split '{args.split}' or dataset config error for '{dataset_name}': {e}. Skipping.")
             continue
        except Exception as e:
            logging.error(f"Failed to load dataset '{dataset_name}' split '{args.split}': {e}. Skipping.")
            continue

        if not args.streaming and dataset:
             if args.column_name not in dataset.column_names:
                 logging.error(f"Column '{args.column_name}' not found in dataset '{dataset_name}' split '{args.split}'. Skipping.")
                 logging.info(f"Available columns: {dataset.column_names}")
                 continue

        logging.info(f"Calculating difficulty scores for column '{args.column_name}' in {dataset_name}...")
        difficulty_scores = []
        try:
            if hasattr(get_difficulty_scores, "warned"):
                delattr(get_difficulty_scores, "warned")

            mapped_dataset = dataset.map(
                get_difficulty_scores,
                fn_kwargs={"column_name": args.column_name},
                batched=True,
                batch_size=args.batch_size,
                remove_columns=dataset.column_names if not args.streaming else None
            )

            logging.info(f"Collecting scores for {dataset_name}...")
            if args.streaming:
                 temp_scores = []
                 for batch in tqdm(mapped_dataset, desc=f"Processing {dataset_name}"):
                     if 'difficulty' in batch and isinstance(batch['difficulty'], list):
                          temp_scores.extend(batch['difficulty'])
                 difficulty_scores = temp_scores
            else:
                difficulty_scores = mapped_dataset["difficulty"]

        except KeyError:
            logging.error(f"Column '{args.column_name}' caused a KeyError during processing for {dataset_name}. Skipping.")
            if not args.streaming and dataset:
                 try:
                     logging.info(f"Available columns: {dataset.column_names}")
                 except Exception as e_inner:
                     logging.warning(f"Could not retrieve column names: {e_inner}")
            continue
        except Exception as e:
            logging.error(f"Error calculating scores for {dataset_name}: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            continue

        if not difficulty_scores:
            logging.warning(f"No difficulty scores were calculated for {dataset_name}. Check dataset, column name, and potential errors.")
            continue

        all_difficulty_scores[dataset_name] = difficulty_scores

        logging.info(f"Successfully calculated {len(difficulty_scores)} scores for {dataset_name}.")
        min_score = np.min(difficulty_scores)
        max_score = np.max(difficulty_scores)
        avg_score = np.mean(difficulty_scores)
        median_score = np.median(difficulty_scores)
        logging.info(f"Stats for {dataset_name}: Min: {min_score}, Max: {max_score}, Avg: {avg_score:.2f}, Median: {median_score}")


    if not all_difficulty_scores:
        logging.error("No data to plot. Exiting.")
        sys.exit(1)

    logging.info("Plotting histogram...")
    plt.figure(figsize=(12, 7))

    all_scores_flat = [score for scores in all_difficulty_scores.values() for score in scores]
    if not all_scores_flat:
        logging.error("No scores found across all datasets. Cannot generate plot.")
        sys.exit(1)

    # Define bins to be centered on integers, e.g., for score 1, bin is [0.5, 1.5]
    min_val_all = int(np.min(all_scores_flat))
    max_val_all = int(np.max(all_scores_flat))
    bins = np.arange(min_val_all - 0.5, max_val_all + 1.5, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_difficulty_scores)))

    for i, (dataset_name, scores) in enumerate(all_difficulty_scores.items()):
        plt.hist(scores, bins=bins, alpha=0.6, label=dataset_name.replace('/', '_'), color=colors[i], edgecolor='black')

    # --- Add Cumulative Percentage Lines and Stats (if single dataset) ---
    if len(all_difficulty_scores) == 1:
        scores = next(iter(all_difficulty_scores.values()))
        scores_arr = np.array(scores)

        # --- Cumulative Percentage Lines ---
        if len(scores_arr) > 0:
            min_val = int(np.min(scores_arr))
            max_val = int(np.max(scores_arr))
            total_count = len(scores_arr)

            logging.info("Calculating cumulative percentage for each integer score...")
            y_max = plt.gca().get_ylim()[1]
            label_y_pos = y_max * 0.95

            integers_to_plot = range(min_val, max_val + 1)
            num_integers = len(integers_to_plot)

            # If the integer range is too large, plot for a subset to avoid clutter
            if num_integers > 20:
                step = round(num_integers / 20) or 1
                integers_to_plot = list(integers_to_plot)[::step]
                logging.info(f"Range of scores is large ({num_integers}). Plotting for a subset of integers.")

            for val in integers_to_plot:
                cumulative_count = np.sum(scores_arr <= val)
                percentage = (cumulative_count / total_count) * 100
                
                # Plot line after the integer value
                line_pos = val + 0.5
                plt.axvline(line_pos, color='green', linestyle='dashed', linewidth=1)
                
                # Add text label showing cumulative percentage
                plt.text(line_pos + 0.1, label_y_pos, f'{percentage:.1f}%', color='green', verticalalignment='top', fontsize=7)

        # --- Stats Text ---
        min_score = np.min(scores)
        max_score = np.max(scores)
        avg_score = np.mean(scores)
        median_score = np.median(scores)

        stats_text = f'Min: {min_score}\nMax: {max_score}\nAvg: {avg_score:.2f}\nMedian: {median_score}'
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title(f'Distribution of Difficulty Scores for "{args.column_name}" column\nDatasets: {", ".join(args.dataset_names)} ({args.split} split)')
    plt.xlabel("Difficulty Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.legend()

    plot_filename = args.plot_filename or f"{'_'.join(d.replace('/', '_') for d in args.dataset_names)}_{args.split}_{args.column_name}_difficulty_dist.png"
    try:
        plt.savefig(plot_filename)
        logging.info(f"Histogram saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot to {plot_filename}: {e}")

    if args.show_plot:
        logging.info("Displaying plot...")
        plt.show()


# --- Argument Parsing and Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot difficulty score distribution for Hugging Face datasets.')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, help='One or more Hugging Face dataset names.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use (e.g., "train", "validation").')
    parser.add_argument('--column_name', type=str, default='difficulty', help='Name of the column containing integer difficulty scores (default: "difficulty").')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for mapping function (default: 1000).')
    parser.add_argument('--num_bins', type=int, default=50, help='Number of bins for the histogram (default: 50).')
    parser.add_argument('--plot_filename', type=str, default=None, help='Filename to save the plot. Defaults to D1_D2_..._SPLIT_COL_difficulty_dist.png')
    parser.add_argument('--streaming', action='store_true', help='Load dataset in streaming mode.')
    parser.add_argument('--max_samples_streaming', type=int, default=0, help='Max samples to process in streaming mode (0 for all).')
    parser.add_argument('--show_plot', action='store_true', help='Display the plot after saving.')

    args = parser.parse_args()

    if args.streaming and args.max_samples_streaming < 0:
        logging.error("--max_samples_streaming cannot be negative.")
        sys.exit(1)
    elif not args.streaming and args.max_samples_streaming != 0:
        logging.warning("--max_samples_streaming is ignored when --streaming is not used.")

    main(args) 