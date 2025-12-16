"""
OpenAI Batch Job Results Processor and Dataset Uploader for stdio format

This script processes completed OpenAI batch job results for code generation tasks
that transform problems into a standard input/output (stdio) format. It retrieves batch
outputs, validates the generated code solutions against test cases using a local
execution service, and uploads the validated results to the HuggingFace Hub.

This is part of a two-step process where batch jobs are first created
(e.g., by `code_create_batch_stdio.py`), then processed with this script.

Features:
- Retrieves and processes results from multiple OpenAI batch jobs.
- Validates generated code solutions by executing them against test cases via a local API.
- Filters results, keeping only solutions that pass at least 80% of their test cases.
- Uploads the successful, validated transformations to a specified HuggingFace Hub repository.
- Extracts a unique ID from the batch job response to map results correctly.

Prerequisites:
1. Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).
2. The batch jobs must be in "completed" status.
3. The local code execution API service must be running. It can be started with Docker:
   ```bash
   # In the repository root directory
   docker build -t code-api -f open_instruct/code/Dockerfile .
   docker run -p 1234:1234 code-api
   ```

Usage:
    python code_upload_batch_stdio.py <batch_ids_json_filepath>

Arguments:
    batch_ids_json_filepath: The path to a JSON file containing a list of OpenAI batch job IDs to process.

Process:
1.  Checks the status of each specified batch job.
2.  Downloads and parses the results from all completed batch jobs.
3.  For each result, it sends the generated solution and its test cases to the local code execution API for validation.
4.  It keeps only the solutions that meet the pass rate threshold.
5.  The script uploads the filtered, high-quality results to the HuggingFace Hub.

Output:
The script updates the target HuggingFace dataset with validated code solutions. Each uploaded example includes:
- `messages`: The user prompt in chat format (the rewritten problem description).
- `ground_truth`: The test cases that passed validation.
- `dataset`: An identifier for the dataset ("code_stdio").
- `good_program`: A quality flag from the generation process.
- `rewritten_solution`: The validated stdio-based code solution.
- `rewritten_input`: The stdio-based problem description.
"""

import hashlib
import json
import os
import sys
import traceback

import requests
from datasets import Dataset
from openai import AzureOpenAI
from pydantic import BaseModel

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

OUTPUT_HF_DATASET = "saurabh5/open-code-reasoning-rlvr-2"
SPLIT = "python"


def get_input(row):
    return row["input"][0]["content"]


def get_id(row):
    input_str = get_input(row)
    if input_str:
        id = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        return id
    return None


def extract_id_from_custom_id(custom_id: str) -> str:
    # get rid of the timestamp
    if "_" in custom_id:
        return "_".join(custom_id.split("_")[1:])
    return custom_id


class OpenAIStructuredOutput(BaseModel):
    rewritten_input: str
    rewritten_solution: str
    test_cases: list[str]
    good_program: bool


def check_batch_status(batch_id: str) -> bool:
    """Check if the batch job is complete."""
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch.status}")
        if batch.status != "completed":
            print(batch)
        return batch.status == "completed"
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return False


def get_batch_results(batch_id: str) -> list:
    """Retrieve and process batch results."""
    try:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(f"Batch {batch_id} is not complete. Current status: {batch.status}")
            return []

        # Download the output file
        output_file = client.files.retrieve(batch.output_file_id)
        output_content = client.files.content(output_file.id)

        # Get the text content from the response
        content_str = output_content.text

        # Process each line of the output
        results = []
        for line in content_str.split("\n"):
            if line.strip():
                try:
                    result = json.loads(line)
                    if "response" in result and "body" in result["response"]:
                        content = result["response"]["body"]["choices"][0]["message"]["content"]
                        processed_result = json.loads(content)
                        # Extract the original ID from custom_id (format: TIMESTAMP_ID)
                        custom_id = result.get("custom_id", "")
                        processed_result["id"] = extract_id_from_custom_id(custom_id)
                        results.append(processed_result)
                except Exception as e:
                    print(f"Error processing result line: {e}")

        return results
    except Exception as e:
        print(f"Error retrieving batch results: {e}")
        return []


def process_batch_results(batch_ids: list[str]):
    """Process the batch results and upload to Hugging Face."""
    all_batch_results = []
    for batch_id in batch_ids:
        if not check_batch_status(batch_id):
            print(f"Batch {batch_id} is not complete yet, skipping")
            continue

        batch_results_single = get_batch_results(batch_id)
        if not batch_results_single:
            print(f"No results found in batch {batch_id}")
            continue
        all_batch_results.extend(batch_results_single)

    if not all_batch_results:
        print("No results found in any of the batches.")
        return

    print(f"Total results from all batches: {len(all_batch_results)}")
    print(f"Sample result: {all_batch_results[0]}")

    # Filter and validate results
    url = "http://localhost:1234/test_program_stdio"
    new_results = []
    # original_dataset = load_dataset(INPUT_HF_DATASET, "SFT", split=SPLIT, num_proc=open_instruct_utils.max_num_processes())

    # Create a lookup dictionary for O(1) access
    print("here")
    # id_to_row = {get_id(row): row for row in original_dataset}
    print("created id_to_row")
    for result in all_batch_results:
        try:
            # Look up the row directly using the ID
            # if result['id'] not in id_to_row:
            #    print(f"No matching row found for ID: {result['id']}")
            #    continue
            # original_dataset_row = id_to_row[result['id']]

            test_cases_raw = result["test_cases"]
            if isinstance(test_cases_raw, list) and len(test_cases_raw) > 0 and isinstance(test_cases_raw[0], str):
                test_cases = [json.loads(tc) for tc in test_cases_raw]
            else:
                test_cases = test_cases_raw

            rewritten_solution = result["rewritten_solution"]

            # Test data
            payload = {"program": rewritten_solution, "tests": test_cases, "max_execution_time": 6.0}

            # Send POST request
            response = requests.post(url, json=payload)
            response_json = response.json()

            if response.ok and sum(response_json["results"]) >= 0.8 * len(test_cases):
                # Keep only passed test cases
                passed_test_cases = [test_cases[j] for j in range(len(test_cases)) if response_json["results"][j] == 1]

                new_results.append(
                    {
                        # **original_dataset_row,
                        "messages": [{"role": "user", "content": result["rewritten_input"]}],
                        # "original_input": get_input(original_dataset_row),
                        "ground_truth": json.dumps(passed_test_cases),
                        "dataset": "code_stdio",
                        "good_program": result["good_program"],
                        "rewritten_solution": rewritten_solution,
                        "rewritten_input": result["rewritten_input"],
                    }
                )
            else:
                print(f"Not adding. Test results: {response_json}")
        except Exception as e:
            print(f"Error processing result: {e}")
            print("test_cases_raw", test_cases_raw)
            print(traceback.format_exc())

    print(
        f"After filtering, {len(new_results)} results out of {len(all_batch_results)} remain. Do you want to upload?"
    )
    breakpoint()

    # Upload to Hugging Face
    if new_results:
        dataset = Dataset.from_list(new_results)
        dataset.push_to_hub(OUTPUT_HF_DATASET)
        print(f"Uploaded {len(new_results)} examples to {OUTPUT_HF_DATASET}")
    else:
        print("No valid results to upload")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python open_code_reasoning_upload_batch.py <batch_ids_json_filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    try:
        with open(filepath) as f:
            batch_ids = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)

    if not isinstance(batch_ids, list):
        print("Error: JSON file should contain a list of batch IDs.")
        sys.exit(1)

    process_batch_results(batch_ids)
