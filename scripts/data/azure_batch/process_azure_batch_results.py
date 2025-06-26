#!/usr/bin/env python3
"""
Process Azure OpenAI batch results, update completions, and (optionally) push a
new Hugging Face dataset that keeps the original prompts **or save to a local
JSONL file**.

Example (push to the Hub):
    python process_azure_batch_results.py <batch_id> \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-hf-username/new-dataset \
        --split split_0

Example (local files):
    python process_azure_batch_results.py batch1,batch2 \
        --input-dataset data/my_prompts.jsonl \
        --output-dataset data/my_updated.jsonl
"""
import argparse
import copy
import json
import os
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import datasets
import openai
import requests
from regenerate_dataset_completions import MODEL_COSTS_PER_1M_TOKENS

###############################################################################
# Utility helpers for flexible dataset I/O (Hub *or* local JSONL)             #
###############################################################################

def _is_jsonl_path(p: str) -> bool:
    """Return True if *p* looks like a local JSON Lines file path."""
    return p.lower().endswith(".jsonl")


def load_input_dataset(path_or_name: str, split: str):
    """Load a dataset either from the Hugging Face Hub or from a local JSONL.

    * If *path_or_name* ends with ``.jsonl`` **and** the file exists locally,
      it is loaded via :pyfunc:`datasets.load_dataset` with the ``json``
      builder.
    * Otherwise the argument is treated as a Hub dataset name (optionally with
      namespace) and loaded in the usual way.
    """
    if _is_jsonl_path(path_or_name) and os.path.isfile(path_or_name):
        print(f"[data] Loading local JSONL dataset from {path_or_name}")
        # The JSON builder creates the default split named "train"
        return datasets.load_dataset("json", data_files=path_or_name, split="train")

    print(f"[data] Loading Hub dataset {path_or_name}:{split}")
    return datasets.load_dataset(path_or_name, split=split)


def save_output_dataset(ds: datasets.Dataset, dest: str, split: str, push: bool):
    """Persist *ds* either to *dest* (local JSONL) or push to the Hub.

    The decision is based on whether *dest* ends with ``.jsonl``.
    """
    if _is_jsonl_path(dest):
        # Always save – --no-upload is ignored for local files because the user
        # explicitly provided a path.
        print(f"[data] Saving dataset locally to {dest}")
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(dest, orient="records", lines=True)
        print(f"[data] Wrote {len(ds):,} rows to {dest}")
        return

    # Otherwise treat *dest* as a Hub dataset repository string.
    if push:
        print(f"[data] Pushing dataset to the Hugging Face Hub → {dest}:{split}")
        token = os.environ.get("HF_TOKEN")
        ds.push_to_hub(dest, split=split, token=token)
        print("[data] Push complete")
    else:
        print("--no-upload specified; dataset not pushed.")

###############################################################################
# Original script (lightly modified to use the helpers above)                 #
###############################################################################

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
        description="Create a new dataset with updated completions from an Azure "
        "OpenAI batch run. *input-dataset* and *output-dataset* can now be either "
        "a Hugging Face dataset name or a local .jsonl file path."
    )
    p.add_argument(
        "batch_id",
        help="Azure batch job ID(s) to process (comma‑separated for multiple)",
    )
    p.add_argument(
        "--input-dataset",
        default="allenai/tulu-3-sft-personas-code",
        help="Source HF dataset *or* local .jsonl containing the original prompts",
    )
    p.add_argument(
        "--output-dataset",
        default="finbarr/tulu-3-sft-personas-code-o3",
        help="Target HF dataset repo *or* local .jsonl file for the updated rows",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name in the source dataset (ignored for local JSONL files)",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip pushing to the Hub (dry‑run); ignored if output is local JSONL",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        help="Limit the number of rows in the output dataset (for testing)",
    )
    return p.parse_args()


client = openai.AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview",
)


def extract_id_from_custom_id(custom_id: str) -> str:
    return "_".join(custom_id.split("_", 1)[1:]) if "_" in custom_id else custom_id


def check_batch_status(batch_id: str) -> bool:
    try:
        return client.batches.retrieve(batch_id).status == "completed"
    except Exception as e:
        print(f"[batch status] {e}")
        return False


def get_batch_results(batch_id: str) -> Tuple[dict[str, str], TokenUsage]:
    """Return mapping result_id → content and aggregate token statistics."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} not complete (status={batch.status}).")

    output_file = client.files.retrieve(batch.output_file_id)
    content_str = client.files.content(output_file.id).text

    results: dict[str, BatchResult] = {}
    total_prompt_tokens = total_completion_tokens = total_tokens = 0

    for line in content_str.splitlines():
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from line: {e}")
            continue

        result_id = extract_id_from_custom_id(result["custom_id"])

        # Content filtering
        choice = result["response"]["body"]["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            filtered_categories = [
                f"{cat} ({details.get('severity', 'unknown')})"
                for cat, details in choice.get("content_filter_results", {}).items()
                if details.get("filtered")
            ]
            print(
                f"Content filtered for {result_id}. Filtered categories: {', '.join(filtered_categories)}"
            )
            continue

        content = choice["message"]["content"]

        usage = result["response"]["body"]["usage"]
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

        results[result_id] = BatchResult(result_id=result_id, content=content)

    token_usage = TokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )
    return results, token_usage


def download_file(file_id: str, dest: pathlib.Path) -> None:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version=2024-07-01-preview"
    response = requests.get(
        url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=120
    )
    response.raise_for_status()
    dest.write_bytes(response.content)


def load_jsonl(file_path: pathlib.Path) -> list[dict]:
    with file_path.open() as f:
        return [json.loads(line) for line in f]


def process_single_batch(batch_id: str, id_lookup: dict) -> Tuple[dict[str, BatchResult], TokenUsage]:
    if not check_batch_status(batch_id):
        print(f"Batch {batch_id} not complete; skipping.")
        return {}, TokenUsage(0, 0, 0)

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/batches/{batch_id}?api-version=2024-07-01-preview"
    r = requests.get(url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=30)
    r.raise_for_status()
    job = r.json()

    # Display up to a handful of errors (if any)
    if job.get("error_file_id"):
        error_file = pathlib.Path(".errors.jsonl")
        try:
            download_file(job["error_file_id"], error_file)
            errors = load_jsonl(error_file)
            print(f"\nBatch {batch_id} Errors: {len(errors)}")
            print("-" * 80)
            for i, error in enumerate(errors[:4]):
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
    split: str,
    push: bool,
    max_rows: Optional[int] = None,
):
    # ------------------------------------------------------------------
    # Load source dataset (Hub or local JSONL)
    # ------------------------------------------------------------------
    original_ds = load_input_dataset(input_dataset, split)
    id_lookup = {row["id"]: row for row in original_ds}

    all_batch_results: dict[str, BatchResult] = {}
    total_token_usage = TokenUsage(0, 0, 0)

    # ------------------------------------------------------------------
    # Aggregate results across batches
    # ------------------------------------------------------------------
    for batch_id in batch_ids:
        batch_results, token_usage = process_single_batch(batch_id, id_lookup)
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

    # ------------------------------------------------------------------
    # Build updated dataset rows
    # ------------------------------------------------------------------
    new_rows = []
    for result_id, batch_result in all_batch_results.items():
        row = id_lookup.get(result_id)
        if row is None:
            print(f"[skip] id {result_id} not in source dataset")
            continue
        missing_features = set(original_ds.features.keys()) - set(row.keys())
        if missing_features:
            raise ValueError(f"Row {result_id} missing features: {missing_features}")

        updated = copy.deepcopy(row)
        assert len(updated["messages"]) == 2, "Expected exactly two messages (user + assistant)"
        updated["messages"][1]["content"] = batch_result.content
        new_rows.append(updated)

    print(f"Kept {len(all_batch_results)} examples from batches.")
    print(f"Previous dataset had {len(original_ds)} examples.")

    if abs(len(all_batch_results) - len(original_ds)) > 0.01 * len(original_ds):
        raise ValueError(
            "New dataset row count differs by more than 1% from the original dataset."
        )

    # ------------------------------------------------------------------
    # Create the output Dataset object
    # ------------------------------------------------------------------
    ds_out = datasets.Dataset.from_list(new_rows, features=original_ds.features)

    # Optional row limit for quick testing
    if max_rows is not None:
        ds_out = ds_out.select(range(min(max_rows, len(ds_out))))
        print(f"Limited output dataset to {len(ds_out)} rows (max_rows={max_rows})")

    # Sanity‑check features match
    if ds_out.features != original_ds.features:
        raise ValueError(
            "Feature mismatch between original and new dataset – aborting save/push."
        )

    # ------------------------------------------------------------------
    # Persist the output dataset (local JSONL or Hub push)
    # ------------------------------------------------------------------
    save_output_dataset(ds_out, output_dataset, split, push)


def main():
    args = parse_args()
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
