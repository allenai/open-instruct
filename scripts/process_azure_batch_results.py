#!/usr/bin/env python3
"""
Process Azure OpenAI batch results, update completions, and (optionally) push a
new Hugging Face dataset that keeps the original prompts.

Example:
    python open_code_reasoning_upload_batch.py <batch_id> \
        --input-dataset nvidia/OpenCodeReasoning \
        --output-dataset your-hf-username/new-dataset \
        --split split_0
"""
import argparse
import copy
import json
import os

import datasets
import openai

# ------------- CLI -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a new dataset with updated completions "
        "from an Azure OpenAI batch run."
    )
    p.add_argument("batch_id", help="Azure batch job ID to process")
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
    p.add_argument(
        "--split",
        default="train",
        help="Split name in the source dataset (and for the output dataset)",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip pushing to the Hub (dry-run)",
    )
    return p.parse_args()


# ------------- Helpers -------------------------------------------------------

client = openai.AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview"
)


def extract_id_from_custom_id(custom_id: str) -> str:
    return "_".join(custom_id.split("_")[1:]) if "_" in custom_id else custom_id


def check_batch_status(batch_id: str) -> bool:
    try:
        return client.batches.retrieve(batch_id).status == "completed"
    except Exception as e:
        print(f"[batch status] {e}")
        return False


def get_batch_results(batch_id: str) -> dict[str, str]:
    """Returns a dictionary of result_id -> content.

    Args:
        batch_id: The ID of the batch job to process.

    Raises:
        ValueError: If the batch job is not complete.

    Returns:
        A dictionary of result_id -> content.
    """
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} not complete (status={batch.status}).")

    output_file = client.files.retrieve(batch.output_file_id)
    content_str = client.files.content(output_file.id).text

    results = {}
    for line in content_str.splitlines():
        print(f'{line=}')
        if not line.strip():
            continue
        result = json.loads(line)
        result_id = extract_id_from_custom_id(result["custom_id"])
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        print(f'{content=}')
        results[result_id] = content
    return results


# ------------- Main processing ----------------------------------------------


def process_batch_results(
    *,
    batch_id: str,
    input_dataset: str,
    output_dataset: str,
    split: str,
    push: bool,
):
    if not check_batch_status(batch_id):
        print("Batch not complete; aborting.")
        return

    batch_results = get_batch_results(batch_id)
    if not batch_results:
        print("No results found.")
        return

    original_ds = datasets.load_dataset(input_dataset, split=split)
    id_lookup = {row["id"]: row for row in original_ds}

    new_rows = []

    for result_id, content in batch_results.items():
        print(f'{result_id=}')
        row = id_lookup[result_id]
        if row is None:
            print(f"[skip] id {result_id=} not in source dataset")
            continue

        # --- build updated example -----------------------------------------
        updated = copy.deepcopy(row)

        # We should always have two messages: user and assistant.
        assert len(updated["messages"]) == 2
        updated["messages"][1]["content"] = batch_results[result_id]
        updated["dataset"] = output_dataset

        new_rows.append(updated)

    print(f"Kept {len(new_rows)} / {len(batch_results)} examples.")

    if not new_rows:
        return

    ds_out = datasets.Dataset.from_list(new_rows)
    if push:
        token = os.environ.get("HF_TOKEN")
        ds_out.push_to_hub(output_dataset, split=split, token=token)
        print(f"Pushed to {output_dataset} ({split})")
    else:
        print("--no-upload specified; dataset not pushed.")


# ------------- Entry point ---------------------------------------------------


def main():
    args = parse_args()
    process_batch_results(
        batch_id=args.batch_id,
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        split=args.split,
        push=not args.no_upload,
    )


if __name__ == "__main__":
    main()