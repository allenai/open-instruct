"""This script is used to regenerate completions for a dataset using a specific OpenAI model.

Usage:

Cd into the directory of this file and run:
```
python regenerate_dataset_completions.py [--sample-limit SAMPLE_LIMIT] \
    [--input-dataset INPUT_DATASET] [--split SPLIT] [--model MODEL] \
    [--dry-run]
```

"""

import argparse
import dataclasses
import json
import os
import random
import sys
import time

import datasets
import openai
import tiktoken

# Model costs are in USD per million tokens.
MODEL_COSTS_PER_1M_TOKENS = {
    "o3-batch": {"input": 10 * 0.5, "output": 40 * 0.5},
    # Annoyingly, this is actually the same as o3-batch, but it varies by region.
    "o3": {"input": 10 * 0.5, "output": 40 * 0.5},
}

# Maximum number of prompts per batch file
MAX_PROMPTS_PER_BATCH = 95_000


@dataclasses.dataclass
class PromptData:
    id: str
    prompt: str


def create_batch_files(
    prompts: list[PromptData],
    base_batch_file_name: str,
    model: str,
    timestamp: int,
    max_completion_tokens: int,
    add_reasoning_summary: bool,
    tool_choice: str,
) -> list[str]:
    """Create multiple batch files in the format required by Azure OpenAI Batch API.
    Returns a list of created batch file paths."""
    batch_file_paths = []

    # Split prompts into chunks of MAX_PROMPTS_PER_BATCH
    for i in range(0, len(prompts), MAX_PROMPTS_PER_BATCH):
        chunk = prompts[i : i + MAX_PROMPTS_PER_BATCH]
        batch_file_name = f"{base_batch_file_name}_{i//MAX_PROMPTS_PER_BATCH}.jsonl"

        with open(batch_file_name, "w") as f:
            for prompt in chunk:
                # Format each request according to batch API requirements
                batch_request = {
                    "custom_id": f"{timestamp}_{prompt.id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt.prompt}],
                        "max_completion_tokens": max_completion_tokens,
                    },
                }
                f.write(json.dumps(batch_request) + "\n")

        batch_file_paths.append(batch_file_name)

    return batch_file_paths


def find_cached_results(id: str, response_dir: str) -> dict | None:
    all_files = []
    for root, _, files in os.walk(response_dir):
        for file in files:
            if file.endswith(f"openai_response_{id}.json"):
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if not all_files:
        return None

    with open(all_files[0]) as f:
        try:
            response = json.load(f)
            rewritten_input = response["rewritten_input"]
            if isinstance(rewritten_input, dict):
                return None
            return response
        except Exception:
            return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate completions for a dataset using a specific OpenAI model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset configuration group
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--input-dataset",
        type=str,
        default="allenai/tulu-3-sft-personas-code",
        help="Name of the input dataset (default: allenai/tulu-3-sft-personas-code)",
    )
    dataset_group.add_argument("--split", type=str, default="train", help='Dataset split to use (default: "train")')
    dataset_group.add_argument(
        "--sample-limit",
        type=int,
        help="Limit the number of samples to process. If not specified, processes all samples.",
    )
    dataset_group.add_argument(
        "--add-reasoning-summary", action="store_true", help="Add reasoning summary to the completions"
    )
    dataset_group.add_argument(
        "--tool-choice",
        type=str,
        default="auto",
        choices=["auto", "none", "required"],
        help='Tool choice to use for the completions (default: "auto")',
    )
    dataset_group.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="Maximum number of completion tokens to use (default: 8192)",
    )

    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="o3-batch",
        choices=list(MODEL_COSTS_PER_1M_TOKENS.keys()),
        help=f'Model to use for completions. Available models: {", ".join(MODEL_COSTS_PER_1M_TOKENS.keys())}',
    )

    # Runtime options group
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--dry-run", action="store_true", help="Preview what would happen without making any API calls"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be a positive integer")

    return args


def main(
    sample_limit: int | None = None,
    input_dataset_name: str = "allenai/tulu-3-sft-personas-code",
    split: str = "split_0",
    model: str = "o3-batch",
    current_dir: str | None = None,
    dry_run: bool = False,
    max_completion_tokens: int = 8192,
    add_reasoning_summary: bool = False,
    tool_choice: str = "auto",
) -> None:
    current_dir = current_dir or os.getcwd()
    timestamp = int(time.time())
    base_batch_file_name = f"{current_dir}/batch_files/{timestamp}"

    # Make sure that the batch files directory exists.
    os.makedirs(f"{current_dir}/batch_files", exist_ok=True)

    print(f"Loading dataset {input_dataset_name} with split {split}")
    input_dataset = datasets.load_dataset(input_dataset_name, split=split, num_proc=max_num_processes())

    # First get all unique IDs
    print(f"Processing dataset with {len(input_dataset)} rows")
    unique_rows = list({row["id"]: row for row in input_dataset}.values())
    print(f"Found {len(unique_rows)} unique rows out of {len(input_dataset)} total rows.")

    # Now sample from unique rows
    random.seed(42)
    sample_limit = len(unique_rows) if sample_limit is None else sample_limit
    sampled_rows = random.sample(unique_rows, min(sample_limit, len(unique_rows)))

    print(f"Processing {len(sampled_rows)} unique rows")

    prompts: list[PromptData] = []
    tokenizer = tiktoken.encoding_for_model(model)
    input_tokens, output_tokens = 0, 0

    for row in sampled_rows:
        # Create prompt
        prompts.append(PromptData(id=row["id"], prompt=row["prompt"]))

        # Count tokens
        prompt_tokens = len(tokenizer.encode(prompts[-1].prompt))
        output = "".join([message["content"] for message in row["messages"] if message["role"] == "assistant"])
        output_tokens = len(tokenizer.encode(output))
        input_tokens += prompt_tokens
        output_tokens += output_tokens

    estimated_cost = (
        input_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["input"]
        + output_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["output"]
    ) / 1_000_000
    print(
        f"Input tokens: {input_tokens}, output tokens: {output_tokens}. "
        f"Estimated cost: ${estimated_cost:.2f} (assuming # of output "
        "tokens is the same with the new model)."
    )

    if dry_run:
        print("Dry run mode - exiting without making API calls")
        sys.exit(0)

    time.sleep(10)
    print("Waiting 10 seconds to allow you to cancel the script if you don't want to proceed...")

    print(f"Creating batch files for {len(prompts)} prompts...")
    batch_file_paths = create_batch_files(
        prompts, base_batch_file_name, model, timestamp, max_completion_tokens, add_reasoning_summary, tool_choice
    )
    print(f"Created {len(batch_file_paths)} batch files")

    # Initialize the client with your API key
    client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-04-01-preview",
    )

    # Submit the batch jobs
    print("Submitting batch jobs to Azure OpenAI...")
    batch_ids = []

    for batch_file_path in batch_file_paths:
        batch_file = client.files.create(file=open(batch_file_path, "rb"), purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )

        batch_ids.append(batch_job.id)
        print(f"Submitted batch job with ID: {batch_job.id}")

    # Print all batch IDs in the requested format
    print(f"\n{input_dataset_name}: {', '.join(batch_ids)}")
    print("\nYou can check the status of your batch jobs using the IDs above.")


if __name__ == "__main__":
    args = parse_args()
    main(
        sample_limit=args.sample_limit,
        input_dataset_name=args.input_dataset,
        split=args.split,
        model=args.model,
        dry_run=args.dry_run,
    )
