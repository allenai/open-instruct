#!/usr/bin/env python3
"""
Process Azure OpenAI batch results, update completions, and (optionally) push a
new Hugging Face dataset that keeps the original prompts.

Example:
    python process_azure_batch_results.py <batch_id> \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-hf-username/new-dataset \
        --split split_0

    # Can also process multiple batches:
    python process_azure_batch_results.py batch1,batch2,batch3 \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-hf-username/new-dataset \
        --split split_0
"""

import argparse
import copy
import json
import os
import pathlib
from dataclasses import dataclass

import datasets
import openai
import requests
from regenerate_dataset_completions import MODEL_COSTS_PER_1M_TOKENS


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
        description="Create a new dataset with updated completions " "from an Azure OpenAI batch run."
    )
    p.add_argument("batch_id", help="Azure batch job ID(s) to process (comma-separated for multiple)")
    p.add_argument(
        "--input-dataset",
        default="allenai/tulu-3-sft-personas-code",
        help="Source HF dataset containing the original prompts",
    )
    p.add_argument(
        "--output-dataset",
        default="finbarr/tulu-3-sft-personas-code-o3",
        help="Target HF dataset to push the updated rows",
    )
    p.add_argument("--split", default="train", help="Split name in the source dataset (and for the output dataset)")
    p.add_argument("--no-upload", action="store_true", help="Skip pushing to the Hub (dry-run)")
    p.add_argument(
        "--max-rows", type=int, help="Limit the number of rows in the output dataset (for testing purposes)"
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


def get_batch_results(batch_id: str) -> tuple[dict[str, str], TokenUsage]:
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
                    filtered_categories.append(f"{category} ({details.get('severity', 'unknown')})")
            print(f"Content filtered for {result_id}. Filtered categories: {', '.join(filtered_categories)}")
            continue

        content = choice["message"]["content"]

        # Extract token usage from the response
        usage = result["response"]["body"]["usage"]
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

        results[result_id] = BatchResult(result_id=result_id, content=content)

    token_usage = TokenUsage(
        prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=total_tokens
    )

    return results, token_usage


def download_file(file_id: str, dest: pathlib.Path) -> None:
    """Download a file from Azure OpenAI API to the specified destination."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version=2024-07-01-preview"

    response = requests.get(url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=120)
    response.raise_for_status()

    with dest.open("wb") as f:
        f.write(response.content)


def load_jsonl(file_path: pathlib.Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with file_path.open() as f:
        return [json.loads(line) for line in f]


def process_single_batch(batch_id: str, id_lookup: dict) -> tuple[dict[str, BatchResult], TokenUsage]:
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
    r = requests.get(url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=30)
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
                print(f"Error: {error.get('error', {}).get('message', 'Unknown error')}")

                if original_row:
                    print(f"\nOriginal Prompt: {original_row['prompt']}")
                else:
                    print("\nOriginal prompt not found in dataset")
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
    batch_ids: list[str],
    input_dataset: str,
    output_dataset: str,
    split: str,
    push: bool,
    max_rows: int | None = None,
):
    # Load the original dataset first so we can look up failed prompts
    original_ds = datasets.load_dataset(input_dataset, split=split, num_proc=max_num_processes())
    id_lookup = {row["id"]: row for row in original_ds}

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

    new_rows = []

    for result_id, batch_result in all_batch_results.items():
        # Check that row has all required features
        row = id_lookup.get(result_id)
        if row is None:
            print(f"[skip] id {result_id=} not in source dataset")
            continue
        missing_features = set(original_ds.features.keys()) - set(row.keys())
        if missing_features:
            raise ValueError(f"Row {result_id=} missing features: {missing_features}")

        # --- build updated example -----------------------------------------
        updated = copy.deepcopy(row)

        # We should always have two messages: user and assistant.
        assert len(updated["messages"]) == 2
        updated["messages"][1]["content"] = batch_result.content

        new_rows.append(updated)

    print(f"Kept {len(all_batch_results)} examples.")
    print(f"Previous dataset had {len(original_ds)} examples.")
    print(f"New dataset has {len(all_batch_results)} examples.")

    if abs(len(all_batch_results) - len(original_ds)) > 0.01 * len(original_ds):
        raise ValueError(
            f"New dataset has {len(all_batch_results)} examples, but "
            f"original dataset had {len(original_ds)} examples."
            "We automatically reject if there's more than a 1% difference."
        )

    if not new_rows:
        return

    # Create dataset using features from the original dataset
    ds_out = datasets.Dataset.from_list(new_rows, features=original_ds.features)

    # Apply row limit if specified
    if max_rows is not None:
        ds_out = ds_out.select(range(min(max_rows, len(ds_out))))
        print(f"Limited output dataset to {len(ds_out)} rows (max_rows={max_rows})")

    print(f"Previous dataset had {len(original_ds)} examples.")
    print(f"New dataset has {len(ds_out)} examples.")

    # Sanity check that features match
    if ds_out.features != original_ds.features:
        print("Warning: Features of new dataset don't match original dataset!")
        print("Original features:", original_ds.features)
        print("New features:", ds_out.features)
        raise ValueError("Feature mismatch between original and new dataset")

    if push:
        token = os.environ.get("HF_TOKEN")
        ds_out.push_to_hub(output_dataset, split=split, token=token)
        print(f"Pushed to {output_dataset} ({split})")
    else:
        print("--no-upload specified; dataset not pushed.")


def main():
    args = parse_args()
    # Split batch_id into list if comma-separated
    batch_ids = args.batch_id.split(",")
    process_batch_results(
        batch_ids=batch_ids,
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        split=args.split,
        push=not args.no_upload,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
