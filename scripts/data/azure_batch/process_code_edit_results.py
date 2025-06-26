#!/usr/bin/env python3
"""
Process Azure OpenAI batch results and replace the code_column in datasets with generated code.

This script processes batch results from code editing jobs and replaces the original
code_column (as specified in DATASET_COLUMN_MAPPINGS) with the newly generated code.
The modified dataset is pushed to Hugging Face Hub.

Example:
    # Push to Hugging Face Hub
    python process_code_edit_results.py <batch_id> \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-username/modified-dataset \
        --split train

    # Can also process multiple batches:
    python process_code_edit_results.py batch1,batch2,batch3 \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-username/modified-dataset \
        --split train

    # Limit output dataset size (for testing):
    python process_code_edit_results.py <batch_id> \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-username/modified-dataset \
        --split train \
        --max-rows 1000
"""

import argparse
import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import datasets
import openai
import requests

# Import DATASET_COLUMN_MAPPINGS from batch_code_edit.py
from batch_code_edit import DATASET_COLUMN_MAPPINGS
from regenerate_dataset_completions import MODEL_COSTS_PER_1M_TOKENS

# Set up logging with file name and line number
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost(self) -> float:
        cost = (
            self.prompt_tokens * MODEL_COSTS_PER_1M_TOKENS["o3-batch"]["input"]
            + self.completion_tokens * MODEL_COSTS_PER_1M_TOKENS["o3-batch"]["output"]
        ) / 1_000_000
        return cost


@dataclass
class BatchResult:
    result_id: str
    content: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process Azure OpenAI batch results and replace the code_column with generated code."
    )
    p.add_argument(
        "batch_id",
        help="Azure batch job ID(s) to process (comma-separated for multiple)",
    )
    p.add_argument(
        "--input-dataset",
        required=True,
        help="Source HF dataset containing the original data",
    )
    p.add_argument(
        "--output-dataset",
        required=True,
        help="Target HF dataset to push the modified dataset",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name in the source dataset",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip pushing to the Hub (dry-run)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        help="Limit the number of rows in the output dataset (for testing purposes)",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation that the number of rows are close to the original dataset",
    )
    return p.parse_args()


client = openai.AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview",
)


def extract_id_from_custom_id(custom_id: str) -> str:
    return "_".join(custom_id.split("_")[1:]) if "_" in custom_id else custom_id


def check_batch_status(batch_id: str) -> bool:
    try:
        return client.batches.retrieve(batch_id).status == "completed"
    except Exception as e:
        print(f"[batch status] {e}")
        return False


def get_batch_results(batch_id: str) -> Tuple[dict[str, str], TokenUsage]:
    """Returns a dictionary of result_id -> content and token usage statistics.

    Args:
        batch_id: The ID of the batch job to process.

    Raises:
        ValueError: If the batch job is not complete.

    Returns:
        A tuple containing:
        - A dictionary of result_id -> content
        - TokenUsage object with token counts and cost
    """
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} not complete (status={batch.status}).")

    output_file = client.files.retrieve(batch.output_file_id)
    content_str = client.files.content(output_file.id).text

    results = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for line in content_str.splitlines():
        if not line.strip():
            continue
        try:
            # Load the JSON line using the default UTF-8 text encoding
            result = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from line: {e}")
            continue
        result_id = extract_id_from_custom_id(result["custom_id"])

        # Check for content filtering
        choice = result["response"]["body"]["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            filtered_categories = []
            for category, details in choice.get("content_filter_results", {}).items():
                if details.get("filtered"):
                    filtered_categories.append(
                        f"{category} ({details.get('severity', 'unknown')})"
                    )
            print(
                f"Content filtered for {result_id}. Filtered categories: {', '.join(filtered_categories)}"
            )
            continue

        content = choice["message"]["content"]

        # Extract token usage from the response
        usage = result["response"]["body"]["usage"]
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

        results[result_id] = BatchResult(
            result_id=result_id,
            content=content,
        )

    token_usage = TokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )

    return results, token_usage


def download_file(file_id: str, dest: pathlib.Path) -> None:
    """Download a file from Azure OpenAI API to the specified destination."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version=2024-07-01-preview"

    response = requests.get(
        url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=120
    )
    response.raise_for_status()

    with dest.open("wb") as f:
        f.write(response.content)


def load_jsonl(file_path: pathlib.Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with file_path.open() as f:
        return [json.loads(line) for line in f]


def process_single_batch(
    batch_id: str, id_lookup: dict
) -> Tuple[dict[str, BatchResult], TokenUsage]:
    """Process a single batch and return its results and token usage.

    Args:
        batch_id: The ID of the batch to process
        id_lookup: Dictionary mapping IDs to original dataset rows

    Returns:
        Tuple containing:
        - Dictionary of result_id -> BatchResult
        - TokenUsage object with token counts and cost
    """
    if not check_batch_status(batch_id):
        print(f"Batch {batch_id} not complete; skipping.")
        return {}, TokenUsage(0, 0, 0)

    # Get batch details to check for errors
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/batches/{batch_id}?api-version=2024-07-01-preview"
    r = requests.get(
        url,
        headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
        timeout=30,
    )
    r.raise_for_status()
    job = r.json()

    # Show errors if they exist
    num_errors = 0
    if job.get("error_file_id"):
        error_file = pathlib.Path(".errors.jsonl")

        try:
            # Download error file
            download_file(job["error_file_id"], error_file)
            errors = load_jsonl(error_file)
            num_errors = len(errors)

            print(f"\nBatch {batch_id} Errors: {num_errors}")
            print("-" * 80)

            for i, error in enumerate(errors):
                if i > 3:
                    break
                request_id = error["custom_id"]
                error_id = extract_id_from_custom_id(request_id)
                original_row = id_lookup.get(error_id)

                print(f"\nError ID: {request_id}")
                print(
                    f"Error: {error.get('error', {}).get('message', 'Unknown error')}"
                )

                if original_row:
                    # Show some information about the original row for debugging
                    print(f"\nOriginal row found in dataset (ID: {error_id})")
                    # Show first few keys to help with debugging
                    row_keys = list(original_row.keys())[:5]
                    print(f"Row keys: {row_keys}")
                else:
                    print(f"\nOriginal row not found in dataset (ID: {error_id})")
                print("-" * 80)

        finally:
            # Clean up temporary files
            error_file.unlink(missing_ok=True)

    batch_results, token_usage = get_batch_results(batch_id)
    if not batch_results:
        print(f"No results found for batch {batch_id}.")
        return {}, TokenUsage(0, 0, 0)

    print(f"\nBatch {batch_id} Token Usage Statistics:")
    print(f"Prompt tokens: {token_usage.prompt_tokens:,}")
    print(f"Completion tokens: {token_usage.completion_tokens:,}")
    print(f"Total tokens: {token_usage.total_tokens:,}")
    print(f"Estimated cost: ${token_usage.cost:.4f}\n")

    return batch_results, token_usage


def process_batch_results(
    *,
    batch_ids: List[str],
    input_dataset: str,
    output_dataset: str,
    split: str = "train",
    max_rows: Optional[int] = None,
    skip_validation: bool = False,
):
    """Process batch results and replace code_column with generated code."""
    # Load the original dataset
    original_ds = datasets.load_dataset(input_dataset, split=split)
    
    # Get column mapping to determine how to create id_lookup
    column_mapping = get_dataset_columns(input_dataset)
    
    # Create id_lookup based on the column mapping
    if column_mapping["id_column"] == "python_index":
        # For python_index, use the row index as the ID
        id_lookup = {str(i): row for i, row in enumerate(original_ds)}
    else:
        # For regular ID columns, use the actual ID value
        id_lookup = {str(row[column_mapping["id_column"]]): row for row in original_ds}

    all_batch_results = {}
    total_token_usage = TokenUsage(0, 0, 0)
    for batch_id in batch_ids:
        batch_results, token_usage = process_single_batch(batch_id, id_lookup)

        # Accumulate results and token usage
        all_batch_results.update(batch_results)
        total_token_usage.prompt_tokens += token_usage.prompt_tokens
        total_token_usage.completion_tokens += token_usage.completion_tokens
        total_token_usage.total_tokens += token_usage.total_tokens

    if not all_batch_results:
        print("No results found in any batch.")
        return

    print("\nTotal Token Usage Statistics:")
    print(f"Prompt tokens: {total_token_usage.prompt_tokens:,}")
    print(f"Completion tokens: {total_token_usage.completion_tokens:,}")
    print(f"Total tokens: {total_token_usage.total_tokens:,}")
    print(f"Estimated cost: ${total_token_usage.cost:.4f}\n")

    # Create a map from custom_id to generated code
    generated_code_map = {}
    for result_id, batch_result in all_batch_results.items():
        # Extract code from the response using the language from the dataset configuration
        print(batch_result.content)
        extracted_code = extract_code_from_response(batch_result.content, column_mapping["language"])
        if extracted_code is not None:
            generated_code_map[result_id] = extracted_code
    
    if not generated_code_map:
        print("No generated code found. Check if batch jobs completed successfully.")
        return
    
    print(f"Extracted {len(generated_code_map)} code snippets from batch results.")
    
    # Modify the dataset with generated code
    modified_dataset = modify_dataset_with_generated_code(
        input_dataset, split, generated_code_map
    )
    
    # Apply row limit if specified
    if max_rows is not None:
        modified_dataset = modified_dataset.select(range(min(max_rows, len(modified_dataset))))
        print(f"Limited output dataset to {len(modified_dataset)} rows (max_rows={max_rows})")
    
    # Push to Hugging Face Hub
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set. Cannot push to Hub.")
        return
    
    modified_dataset.push_to_hub(output_dataset, split=split, token=token)
    print(f"Pushed to {output_dataset} ({split})")


def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get the column mappings for a specific dataset."""
    if dataset_name not in DATASET_COLUMN_MAPPINGS:
        raise ValueError(f"No column mapping found for dataset: {dataset_name}")
    return DATASET_COLUMN_MAPPINGS[dataset_name]


def extract_code_from_response(response_text: str, language: str) -> Optional[str]:
    """Extract code from the API response using regex patterns."""
    # Pattern to match code blocks with language specification
    pattern = rf"```{language}\n(.*?)\n```"
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find any code block
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no code block found, check if response is just 'ERROR'
    if response_text.strip().upper() == "ERROR":
        return None
    
    # If no code block found, return the entire response as code
    return response_text.strip()


def modify_dataset_with_generated_code(
    dataset_name: str,
    split: str,
    generated_code_map: Dict[str, str],
) -> datasets.Dataset:
    """Modify the dataset by replacing the code_column with generated code.
    
    Only includes rows that have been modified with generated code.
    """
    logger.info(f"Modifying dataset {dataset_name}...")
    
    # Load the original dataset
    dataset = datasets.load_dataset(dataset_name, split=split)
    column_mapping = get_dataset_columns(dataset_name)
    
    # Create a new dataset with only modified rows
    modified_rows = []
    
    for i, row in enumerate(dataset):
        # Get the row ID
        row_id = row[column_mapping["id_column"]] if column_mapping["id_column"] != "python_index" else i
        # Convert to string for consistent comparison with generated_code_map keys
        row_id_str = str(row_id)

        # Only include rows that have generated code
        if row_id_str in generated_code_map:
            generated_code = generated_code_map[row_id_str]
            if generated_code is not None:
                # Create a new row with the generated code
                new_row = dict(row)
                new_row[column_mapping["code_column"]] = generated_code
                modified_rows.append(new_row)
    
    # Create new dataset with only modified rows, preserving original features
    try:
        modified_dataset = datasets.Dataset.from_list(modified_rows, features=dataset.features)
    except Exception as e:
        print(f"Error creating modified dataset: {e}")
        print(f"Modified rows: {modified_rows}")
        print(f"Dataset features: {dataset.features}")
        raise e
    
    logger.info(f"Modified dataset created with {len(modified_rows)} rows (only rows with generated code)")
    
    return modified_dataset


def main():
    args = parse_args()
    # Split batch_id into list if comma-separated
    batch_ids = args.batch_id.split(",")
    process_batch_results(
        batch_ids=batch_ids,
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        split=args.split,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
