"""
OpenAI Batch Job Creator for Code Solution Generation

This script creates an OpenAI batch job to generate Python solutions for a given set
of coding problems. It loads problems from a Hugging Face dataset, formats them into
prompts that ask an LLM to provide a solution, and submits them as a batch job to the
Azure OpenAI API.

This script is intended to be the first step in a workflow where solutions are generated
at scale and then processed or validated in a subsequent step.

Features:
- Loads a dataset of coding problems from the HuggingFace Hub.
- Deduplicates problems based on their ID to avoid redundant processing.
- Generates prompts that instruct an LLM to solve the provided problem and return a Python function.
- Creates a `.jsonl` batch file containing all the prompts.
- Submits the batch file to the Azure OpenAI API to start a new batch job.

Prerequisites:
- Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

Usage:
    python code_create_batch_solution.py

Workflow:
1.  Run this script to create and submit a batch job for solution generation.
2.  The script will output a batch job ID.
3.  Wait for the batch job to complete.
4.  The results (generated code solutions) can then be retrieved using the batch job ID for further processing.
"""

import hashlib
import json
import os
import time

from datasets import load_dataset
from openai import AzureOpenAI

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

MODEL = "o3"
SAMPLE_LIMIT = 1000000
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

INPUT_HF_DATASET = "saurabh5/rlvr_acecoder"
SPLIT = "train"


def extract_python_code(model_output: str) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return model_output

    # Return the last match, stripped of whitespace
    return matches[-1].strip()


def get_input(row):
    return row["messages"][0]["content"]


def get_solution(row):
    """
    for message in row['messages']:
        if message['role'] == 'assistant':
            return message['content']
    return None
    """
    return row["solution"]


def get_id(row):
    input = get_input(row)
    if input:
        return hashlib.sha256(input.encode("utf-8")).hexdigest()
    return None


def create_batch_file(prompts):
    """Create a batch file in the format required by Azure OpenAI Batch API."""
    with open(BATCH_FILE_NAME, "w") as f:
        for id, prompt in prompts:
            # Format each request according to batch API requirements
            batch_request = {
                "custom_id": f"{TIMESTAMP}_{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that can write code in Python."},
                        {"role": "user", "content": prompt},
                    ],
                },
            }
            f.write(json.dumps(batch_request) + "\n")


def main():
    global SAMPLE_LIMIT
    input_dataset = load_dataset(INPUT_HF_DATASET, split=SPLIT, num_proc=open_instruct_utils.max_num_processes())

    print(f"Found {len(input_dataset)} total rows")

    # dedeuplicated on id
    unique_ids = set()
    unique_rows = []
    for row in input_dataset:
        if get_id(row) not in unique_ids:
            unique_ids.add(get_id(row))
            unique_rows.append(row)

    sampled_rows = unique_rows

    print(f"Processing {len(sampled_rows)} rows")

    master_prompt = r"""
Your task is to solve coding problems with function-based solutions.

Here is the problem:
<PROBLEM>
{input}
</PROBLEM>

Output should be a valid Python function. Feel free to think step by step before you write the function. Make sure to wrap your solution with ```python and ```.
"""

    prompts = []
    for row in sampled_rows:
        input = get_input(row)
        if input is None:
            continue
        prompts.append((get_id(row), master_prompt.replace("{input}", input)))

    print(f"Creating batch file with {len(prompts)} prompts...")
    print(f"First prompt: {prompts[0]}")
    create_batch_file(prompts)
    print(f"Created batch file at {BATCH_FILE_NAME}")

    # Submit the batch job
    print("Submitting batch job to Azure OpenAI...")
    batch_file = client.files.create(file=open(BATCH_FILE_NAME, "rb"), purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
    )

    print(f"Batch job submitted with ID: {batch_job.id}")
    print("You can check the status of your batch job using the ID above.")


if __name__ == "__main__":
    main()
