"""
OpenAI Batch Job Results Processor and Dataset Uploader

This script processes completed OpenAI batch job results for code generation tasks.
It retrieves batch outputs, validates the generated code solutions against test cases,
and uploads successful results to HuggingFace Hub. This is part of a two-step process
where batch jobs are first created with code_create_batch.py, then processed with this script.

Features:
- Retrieves and processes OpenAI batch job results
- Validates generated code solutions using local execution API
- Filters results based on test case pass rates (≥80% required)
- Uploads successful transformations to HuggingFace Hub
- Handles custom ID extraction from batch job responses
- Provides detailed logging and error handling

Prerequisites:
1. Set up OpenAI API credentials:
   - OPENAI_API_KEY: Your OpenAI API key

2. Complete batch job created by code_create_batch.py:
   - The batch job must be in "completed" status
   - You need the batch job ID from the creation step

3. Start the code execution API:
   ```bash
   # In the repository root directory
   docker build -t code-api -f open_instruct/code/Dockerfile .
   docker run -p 1234:1234 code-api
   ```

Usage:
    python code_upload_batch.py <batch_id>

Arguments:
    batch_id: The OpenAI batch job ID (obtained from code_create_batch.py)

Configuration:
    Modify these variables in the script as needed:
    - INPUT_HF_DATASET: Source HuggingFace dataset
    - OUTPUT_HF_DATASET: Target HuggingFace dataset
    - SPLIT: Dataset split to process

Examples:
    ```bash
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-openai-api-key"

    # Start the code execution API
    docker build -t code-api -f open_instruct/code/Dockerfile .
    docker run -p 1234:1234 code-api

    # Process batch results (replace with your actual batch ID)
    python code_upload_batch.py batch_abc123def456
    ```

Process:
1. Checks if the specified batch job is completed
2. Downloads and parses batch job results
3. Validates each generated solution against its test cases
4. Keeps only solutions that pass ≥80% of test cases
5. Uploads filtered results to HuggingFace Hub
6. Provides statistics on success rates

Output:
    Updates the target HuggingFace dataset with validated code solutions:
    - messages: User prompts in chat format
    - ground_truth: Test cases that passed validation
    - dataset: Dataset identifier
    - good_program: Quality flag from the generation process
    - original_solution/input: Original data from source dataset
    - rewritten_solution/input: Transformed data from batch job

Workflow:
    This script is typically used after code_create_batch.py:
    1. Run code_create_batch.py to create batch job
    2. Wait for batch job to complete (check status periodically)
    3. Run this script with the batch ID to process results

Note:
    The script includes a breakpoint before uploading results, allowing you to
    review the processed data before pushing to HuggingFace Hub.
"""

import json
import os
import sys

import requests
from datasets import Dataset, load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

INPUT_HF_DATASET = "nvidia/OpenCodeReasoning-2"
OUTPUT_HF_DATASET = "saurabh5/open-code-reasoning-rlvr-2"
SPLIT = "python"

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True, num_proc=open_instruct_utils.max_num_processes()),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True, num_proc=open_instruct_utils.max_num_processes()),
    "code_contests": load_dataset("deepmind/code_contests", num_proc=open_instruct_utils.max_num_processes()),
    "open-r1/codeforces": load_dataset("open-r1/codeforces", num_proc=open_instruct_utils.max_num_processes()),
}


def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


def get_input(row):
    ds_name, ds_split, ds_index = row["dataset"], row["split"], int(row["index"])
    if ds_name not in hf_datasets:
        return None
    return get_question(ds_name, ds_split, ds_index)


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


def process_batch_results(batch_id: str):
    """Process the batch results and upload to Hugging Face."""
    if not check_batch_status(batch_id):
        print(f"Batch {batch_id} is not complete yet")
        return

    batch_results = get_batch_results(batch_id)
    if not batch_results:
        print("No results found in batch")
        return

    print(f"Sample result: {batch_results[0]}")

    # Filter and validate results
    url = "http://localhost:1234/test_program"
    new_results = []
    original_dataset = load_dataset(
        INPUT_HF_DATASET, SPLIT, split=SPLIT, num_proc=open_instruct_utils.max_num_processes()
    )

    # Create a lookup dictionary for O(1) access
    id_to_row = {row["id"]: row for row in original_dataset}

    for result in batch_results:
        try:
            # Look up the row directly using the ID
            if result["id"] not in id_to_row:
                print(f"No matching row found for ID: {result['id']}")
                continue
            original_dataset_row = id_to_row[result["id"]]

            test_cases = result["test_cases"]
            rewritten_solution = result["rewritten_solution"]

            # Test data
            payload = {"program": rewritten_solution, "tests": test_cases, "max_execution_time": 2.0}

            # Send POST request
            response = requests.post(url, json=payload)
            response_json = response.json()

            if response.ok and sum(response_json["results"]) >= 0.8 * len(test_cases):
                # Keep only passed test cases
                passed_test_cases = [test_cases[j] for j in range(len(test_cases)) if response_json["results"][j] == 1]

                new_results.append(
                    {
                        **original_dataset_row,
                        "messages": [{"role": "user", "content": result["rewritten_input"]}],
                        "original_input": get_input(original_dataset_row),
                        "ground_truth": passed_test_cases,
                        "dataset": "code",
                        "good_program": result["good_program"],
                        "rewritten_solution": rewritten_solution,
                        "rewritten_input": result["rewritten_input"],
                    }
                )
            else:
                print(f"Not adding. Test results: {response_json}")
        except Exception as e:
            print(f"Error processing result: {e}")

    print(f"After filtering, {len(new_results)} results out of {len(batch_results)} remain. Do you want to upload?")
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
        print("Usage: python open_code_reasoning_upload_batch.py <batch_id>")
        sys.exit(1)

    batch_id = sys.argv[1]
    process_batch_results(batch_id)
