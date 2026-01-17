"""
OpenAI Batch Job Creator for Code Dataset Processing

This script creates OpenAI batch jobs to process coding datasets at scale. It transforms
coding problems into structured formats with function signatures, test cases, and solutions
using Azure OpenAI's batch API. This is the first step in a two-part process, followed by
code_upload_batch.py to process the results.

Features:
- Creates OpenAI batch jobs for large-scale code processing
- Uses structured output with JSON schema validation
- Deduplicates input data by unique IDs
- Supports local caching to avoid reprocessing
- Configurable sampling and processing limits
- Generates properly formatted batch request files

Prerequisites:
1. Set up Azure OpenAI API credentials:
   - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
   - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL

2. Install required packages:
   ```bash
   pip install openai datasets pydantic
   ```

Usage:
    python code_create_batch.py

Configuration:
    Modify these variables in the script as needed:
    - INPUT_HF_DATASET: Source HuggingFace dataset to process
    - SPLIT: Dataset split to use (e.g., "split_0")
    - MODEL: Azure OpenAI model name (e.g., "gpt4.1")
    - SAMPLE_LIMIT: Number of samples to process (None for all)

Output:
    - Creates a batch file in ./batch_files/ directory
    - Submits the batch job to Azure OpenAI
    - Returns a batch job ID for use with code_upload_batch.py

Examples:
    ```bash
    # Set environment variables
    export AZURE_OPENAI_API_KEY="your-api-key"
    export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

    # Run the script
    python code_create_batch.py

    # The script will output a batch job ID like:
    # "Batch job submitted with ID: batch_abc123def456"

    # Use this ID later with code_upload_batch.py:
    # python code_upload_batch.py batch_abc123def456
    ```

Process:
1. Loads and deduplicates the input dataset
2. Samples data according to SAMPLE_LIMIT configuration
3. Generates prompts for code transformation
4. Creates a batch file with structured output schema
5. Submits the batch job to Azure OpenAI
6. Returns batch job ID for result processing

Batch Request Format:
    Each request transforms a coding problem by:
    - Converting input to specify function signatures
    - Rewriting solutions to use function parameters
    - Generating executable test cases
    - Validating program quality

Structured Output Schema:
    - rewritten_input: Function-based problem description
    - rewritten_solution: Function-based solution code
    - test_cases: List of executable assert statements
    - good_program: Boolean quality indicator

Workflow:
    This is typically the first step in a two-part process:
    1. Run this script to create and submit batch job
    2. Wait for batch completion (can take hours for large jobs)
    3. Run code_upload_batch.py with the batch ID to process results

Cost Considerations:
    Batch processing is more cost-effective than individual API calls, but can still
    be expensive for large datasets. Monitor your Azure OpenAI usage and costs.

Note:
    The script includes caching logic to avoid reprocessing previously handled data.
    Cached results are stored locally and checked before creating new batch requests.
"""

import json
import os
import random
import time

from datasets import load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

MODEL = "gpt-4.1"
SAMPLE_LIMIT = 1000000
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

INPUT_HF_DATASET = "nvidia/OpenCodeReasoning-2"
OUTPUT_HF_DATASET = "saurabh5/open-code-reasoning-rlvr-2"
SPLIT = "python"

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True, num_proc=open_instruct_utils.max_num_processes()),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True, num_proc=open_instruct_utils.max_num_processes()),
    "code_contests": load_dataset("deepmind/code_contests", num_proc=open_instruct_utils.max_num_processes()),
    "open-r1/codeforces": load_dataset("open-r1/codeforces", num_proc=open_instruct_utils.max_num_processes()),
}


def extract_python_code(model_output: str) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return model_output

    # Return the last match, stripped of whitespace
    return matches[-1].strip()


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


def get_solution(row):
    """
    for message in row['messages']:
        if message['role'] == 'assistant':
            return message['content']
    return None
    """
    return row["solution"]


def get_id(row):
    return row["question_id"]


class OpenAIStructuredOutput(BaseModel):
    rewritten_input: str
    rewritten_solution: str
    test_cases: list[str]
    good_program: bool


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
                    "max_tokens": 8192,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "OpenAIStructuredOutput",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "rewritten_input": {"type": "string"},
                                    "rewritten_solution": {"type": "string"},
                                    "test_cases": {"type": "array", "items": {"type": "string"}},
                                    "good_program": {"type": "boolean"},
                                },
                                "required": ["rewritten_input", "rewritten_solution", "test_cases", "good_program"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            }
            f.write(json.dumps(batch_request) + "\n")


def find_cached_results(id: str):
    response_dir = f"{WD}/open_ai_responses/"
    all_files = []
    for root, _, files in os.walk(response_dir):
        for file in files:
            if file.endswith(f"openai_response_{id}.json"):
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if all_files:
        with open(all_files[0]) as f:
            try:
                response = json.load(f)
                rewritten_input = response["rewritten_input"]
                if type(rewritten_input) == dict:
                    return None
                return response
            except Exception:
                return None

    return None


def main():
    global SAMPLE_LIMIT
    input_dataset = load_dataset(
        INPUT_HF_DATASET, "train", split=SPLIT, num_proc=open_instruct_utils.max_num_processes()
    )

    # First get all unique IDs
    unique_ids = set()
    unique_rows = []
    for row in input_dataset:
        if row["question_id"] not in unique_ids:
            unique_ids.add(row["question_id"])
            unique_rows.append(row)

    print(f"Found {len(unique_rows)} unique rows out of {len(input_dataset)} total rows")

    # Now sample from unique rows
    random.seed(42)
    if SAMPLE_LIMIT is None:
        SAMPLE_LIMIT = len(unique_rows)
    sampled_rows = random.sample(unique_rows, min(SAMPLE_LIMIT, len(unique_rows)))

    print(f"Processing {len(sampled_rows)} unique rows")

    master_prompt = r"""
# Instructions
Your task is to transform coding problems into a structured dataset format with function-based solutions and test cases.

## Response Rules
- Return only a JSON object with no additional text
- The JSON must include: rewritten_input, rewritten_solution, test_cases, and good_program
- Set good_program to False if the solution is incorrect or the problem is unsuitable

## Transformation Process
1. Convert the input problem to specify a function signature
2. Rewrite the solution to use function parameters instead of input()
3. Create test cases as executable assert statements
4. Package everything in the required JSON format

## Input Requirements
- Keep the rewritten_input similar in length to the original
- Clearly specify the function name and parameters
- Update any examples to use the new function signature

## Test Case Requirements
- Extract test cases from the input when available
- Add new test cases to cover edge cases
- Format as executable Python assert statements
- Do not include comments in test cases

Here is the file input:
<INPUT>
{input}
</INPUT>

Here is a reference solution:
```python
{solution}
```

Output should be a JSON object with this structure:
{
    "rewritten_input": "...",
    "rewritten_solution": "...",
    "test_cases": ["...", "..."],
    "good_program": true/false
}
"""

    prompts = []
    for row in sampled_rows:
        input = get_input(row)
        if input is None:
            continue
        prompts.append((get_id(row), master_prompt.replace("{input}", input).replace("{solution}", get_solution(row))))

    print(f"Creating batch file with {len(prompts)} prompts...")
    print(f"First prompt: {prompts[0]}")
    breakpoint()
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
