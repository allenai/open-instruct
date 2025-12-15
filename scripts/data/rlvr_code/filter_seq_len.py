"""
HuggingFace Dataset Sequence Length Filter

This script filters HuggingFace datasets based on sequence length limits and uploads
the filtered results to a new branch on HuggingFace Hub. It's designed to create
subsets of datasets that fit within specific token length constraints for model training.

Features:
- Filters datasets by maximum sequence length using configurable tokenizers
- Supports both streaming and non-streaming dataset processing
- Uploads filtered datasets to new branches on HuggingFace Hub
- Handles large datasets efficiently with batched processing
- Comprehensive logging and error handling
- Automatic repository and branch creation

Prerequisites:
1. Install required packages:
   ```bash
   pip install datasets transformers huggingface_hub
   ```

2. Login to HuggingFace Hub:
   ```bash
   huggingface-cli login
   ```

Usage:
    python filter_seq_len.py --dataset_name <dataset> --split <split> --column_name <column> --max_seq_len <length> --hub_repo_id <repo> --new_branch_name <branch>

Required Arguments:
    --dataset_name: Source HuggingFace dataset name (e.g., "allenai/c4")
    --split: Dataset split to filter (e.g., "train", "validation")
    --column_name: Column containing text data to check length
    --max_seq_len: Maximum sequence length allowed (inclusive)
    --hub_repo_id: Target HuggingFace repository (e.g., "username/filtered-dataset")
    --new_branch_name: Name of the new branch to create (e.g., "filtered-v1")

Optional Arguments:
    --batch_size: Batch size for processing (default: 1000)
    --streaming: Enable streaming mode for large datasets

Examples:
    # Filter a dataset to maximum 512 tokens
    python filter_seq_len.py \
        --dataset_name "allenai/c4" \
        --split "train" \
        --column_name "text" \
        --max_seq_len 512 \
        --hub_repo_id "myuser/c4-filtered" \
        --new_branch_name "max-512-tokens"

    # Use streaming mode for very large datasets
    python filter_seq_len.py \
        --dataset_name "allenai/c4" \
        --split "train" \
        --column_name "text" \
        --max_seq_len 1024 \
        --hub_repo_id "myuser/c4-filtered" \
        --new_branch_name "max-1024-tokens" \
        --streaming

Process:
1. Loads the specified dataset and tokenizer
2. Applies length filtering in batches
3. Creates target repository and branch if they don't exist
4. Uploads filtered data to the new branch
5. Provides logging throughout the process

Output:
    - Filtered dataset uploaded to specified HuggingFace Hub repository and branch
    - Log messages showing original vs filtered dataset sizes
    - Error messages if any issues occur during processing

Note:
    Ensure you have sufficient permissions to create repositories and branches
    on HuggingFace Hub. The script will create private repositories by default.
"""

import argparse
import json  # For saving streaming data
import os
import sys
import tempfile  # For streaming upload

import datasets
from huggingface_hub import HfApi, HfFolder, create_repo  # For Hub upload
from tqdm import tqdm
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-SFT"


# --- Helper Function for Filtering ---
def check_seq_len_batch(batch, tokenizer, column_name, max_seq_len):
    """
    Tokenizes the specified column for a batch and returns a boolean list
    indicating whether each sequence length is within the limit.
    Designed for use with dataset.map(batched=True).
    """
    if column_name not in batch:
        if not hasattr(check_seq_len_batch, "warned_missing_col"):
            logger.warning(f"Column '{column_name}' not found in a batch. Excluding rows in this batch.")
            check_seq_len_batch.warned_missing_col = True
        # Get number of items in batch from another column
        example_key = next(iter(batch.keys()))
        num_items = len(batch[example_key])
        return {"keep": [False] * num_items}  # Exclude all items in this batch

    outputs = [str(item) if item is not None else "" for item in batch[column_name]]
    try:
        # We only need lengths, no need for truncation/padding if just checking length
        tokenized_outputs = tokenizer(outputs, truncation=False, padding=False)
        lengths = [len(ids) for ids in tokenized_outputs["input_ids"]]
        return {"keep": [length <= max_seq_len for length in lengths]}
    except Exception as e:
        logger.error(f"Error during tokenization/length check in batch: {e}. Excluding rows in this batch.")
        num_items = len(outputs)
        return {"keep": [False] * num_items}  # Exclude all items on error


# --- Main Function ---
def main(args):
    """Loads dataset, filters by sequence length, and saves the result."""
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer {MODEL_NAME}: {e}")
        sys.exit(1)

    logger.info(f"Loading dataset: {args.dataset_name}, split: {args.split}, streaming={args.streaming}")
    try:
        dataset = datasets.load_dataset(
            args.dataset_name, split=args.split, streaming=args.streaming, num_proc=max_num_processes()
        )
    except FileNotFoundError:
        logger.error(f"Dataset '{args.dataset_name}' not found.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid split '{args.split}' or dataset configuration error for '{args.dataset_name}': {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_name}' split '{args.split}': {e}")
        sys.exit(1)

    # --- Filtering --- #
    logger.info(f"Filtering dataset by max sequence length ({args.max_seq_len}) for column '{args.column_name}'...")

    # Reset warning flag for the helper function
    if hasattr(check_seq_len_batch, "warned_missing_col"):
        delattr(check_seq_len_batch, "warned_missing_col")

    # Apply the mapping function to get the 'keep' column
    try:
        dataset_with_keep_flag = dataset.map(
            check_seq_len_batch,
            fn_kwargs={"tokenizer": tokenizer, "column_name": args.column_name, "max_seq_len": args.max_seq_len},
            batched=True,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"Error during the mapping phase for length checking: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Now, filter based on the 'keep' flag
    # Note: dataset.filter() expects a function that returns True/False per example,
    # or a boolean list if batched=True. Since we already computed the boolean
    # list in the 'keep' column via map(), we can leverage that.

    # For non-streaming: filter directly on the mapped dataset
    if not args.streaming:
        logger.info("Applying filter to non-streaming dataset...")
        try:
            filtered_dataset = dataset_with_keep_flag.filter(lambda example: example["keep"])
            # Remove the temporary 'keep' column
            filtered_dataset = filtered_dataset.remove_columns(["keep"])
        except Exception as e:
            logger.error(f"Error applying filter or removing column: {e}")
            sys.exit(1)

        logger.info(
            f"Original dataset size: {len(dataset_with_keep_flag)}"
        )  # Original might be different if map failed batches
        logger.info(f"Filtered dataset size: {len(filtered_dataset)}")

        # --- Uploading --- #
        if args.hub_repo_id:
            # --- Uploading to Hub --- #
            logger.info("Attempting to upload to Hugging Face Hub...")
            hf_token = HfFolder.get_token()
            if not hf_token:
                logger.error("Hugging Face Hub token not found. Please login using `huggingface-cli login`.")
                sys.exit(1)

            # Ensure the repo exists, create if it doesn't (private by default)
            try:
                create_repo(args.hub_repo_id, repo_type="dataset", exist_ok=True, token=hf_token)
                logger.info(f"Ensured repository '{args.hub_repo_id}' exists on the Hub.")
            except Exception as e:
                logger.error(f"Failed to create or access Hub repo '{args.hub_repo_id}': {e}")
                sys.exit(1)

            api = HfApi()

            # Create the target branch before uploading
            try:
                api.create_branch(
                    repo_id=args.hub_repo_id,
                    repo_type="dataset",
                    branch=args.new_branch_name,
                    token=hf_token,
                    exist_ok=True,  # Don't fail if branch already exists
                )
                logger.info(f"Ensured branch '{args.new_branch_name}' exists in repo '{args.hub_repo_id}'.")
            except Exception as e:
                logger.error(f"Failed to create branch '{args.new_branch_name}' in repo '{args.hub_repo_id}': {e}")
                sys.exit(1)

            if not args.streaming:
                # --- Non-Streaming Upload ---
                logger.info(
                    f"Uploading filtered non-streaming dataset to '{args.hub_repo_id}' branch '{args.new_branch_name}' split '{args.split}'..."
                )
                try:
                    filtered_dataset.push_to_hub(
                        args.hub_repo_id,
                        split=args.split,
                        token=hf_token,
                        revision=args.new_branch_name,  # Target the new branch
                    )
                    logger.info("Dataset uploaded successfully to the Hub branch.")
                except Exception as e:
                    logger.error(f"Failed to upload non-streaming dataset to Hub branch '{args.new_branch_name}': {e}")
                    sys.exit(1)
            else:
                # --- Streaming Upload ---
                logger.info(
                    f"Processing stream and uploading to '{args.hub_repo_id}' branch '{args.new_branch_name}' split '{args.split}'..."
                )
                # Write to a temporary JSONL file first
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_f:
                    temp_filename = temp_f.name
                    logger.info(f"Writing filtered stream to temporary file: {temp_filename}")
                    written_count = 0
                    original_count = 0
                    try:
                        for example in tqdm(dataset_with_keep_flag, desc="Filtering stream for upload"):
                            original_count += 1
                            if example.get("keep", False):
                                example.pop("keep", None)
                                temp_f.write(json.dumps(example) + "\n")
                                written_count += 1
                        logger.info(f"Processed approximately {original_count} records.")
                        logger.info(f"Wrote {written_count} filtered records to temporary file.")
                    except Exception as e:
                        logger.error(f"Failed during streaming filtering to temporary file: {e}")
                        os.remove(temp_filename)  # Clean up temp file on error
                        sys.exit(1)

                    # Upload the temporary file
                    if written_count > 0:
                        upload_path_in_repo = f"data/{args.split}.jsonl"  # Standard location
                        logger.info(
                            f"Uploading temporary file {temp_filename} to {args.hub_repo_id} at '{upload_path_in_repo}' on branch '{args.new_branch_name}'..."
                        )
                        try:
                            api.upload_file(
                                path_or_fileobj=temp_filename,
                                path_in_repo=upload_path_in_repo,
                                repo_id=args.hub_repo_id,
                                repo_type="dataset",
                                token=hf_token,
                                revision=args.new_branch_name,  # Target the new branch
                                commit_message=f"Add filtered {args.split} split to branch {args.new_branch_name} (max_len={args.max_seq_len}, col={args.column_name})",
                            )
                            logger.info(
                                f"Successfully uploaded {upload_path_in_repo} to the Hub branch '{args.new_branch_name}'."
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to upload temporary file to Hub branch '{args.new_branch_name}': {e}"
                            )
                        finally:
                            logger.info(f"Cleaning up temporary file: {temp_filename}")
                            os.remove(temp_filename)
                    else:
                        logger.warning("No records were filtered to be kept. Nothing uploaded.")
                        os.remove(temp_filename)  # Still clean up empty temp file
        else:
            # This case should no longer be reachable since hub_repo_id is required
            logger.error("No output method specified (this should not happen).")
            sys.exit(1)


# --- Argument Parsing and Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a Hugging Face dataset based on sequence length and upload to a new branch on the Hub."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help='Name of the source Hugging Face dataset (e.g., "allenai/c4").'
    )
    parser.add_argument("--split", type=str, required=True, help='Dataset split to use (e.g., "train", "validation").')
    parser.add_argument(
        "--column_name", type=str, required=True, help="Name of the column containing text data to check length."
    )
    parser.add_argument("--max_seq_len", type=int, required=True, help="Maximum sequence length allowed (inclusive).")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for filtering (default: 1000).")
    parser.add_argument("--streaming", action="store_true", help="Load and process dataset in streaming mode.")
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        required=True,
        help='Hugging Face Hub repository ID to upload the filtered dataset to (e.g., "username/my-filtered-dataset"). Requires prior `huggingface-cli login`.',
    )
    parser.add_argument(
        "--new_branch_name",
        type=str,
        required=True,
        help='Name of the new branch to create in the Hub repository for the filtered data (e.g., "filtered-v1").',
    )

    args = parser.parse_args()

    # Simple validation
    if "/" not in args.hub_repo_id:
        logger.error("Invalid --hub_repo_id format. Should be 'namespace/repository_name'.")
        sys.exit(1)

    main(args)
