"""
Usage:
Cd into the directory of this file and run:
```
python open_code_reasoning_create_batch.py
```
"""
import json
import os
import time
from openai import AzureOpenAI
from datasets import load_dataset
from pydantic import BaseModel
from typing import List
from collections import Counter
import random
from open_instruct.ground_truth_utils import extract_python_code

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

MODEL = "gpt-4o-mini-batch"
SAMPLE_LIMIT = None
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

INPUT_HF_DATASET = "nvidia/OpenCodeReasoning"
SPLIT = "split_0"

def get_input(row):
    return row['input']

def get_solution(row):
    """
    for message in row['messages']:
        if message['role'] == 'assistant':
            return message['content']
    return None
    """
    return row['solution']

def get_id(row):
    return row['id']

class OpenAIStructuredOutput(BaseModel):
    rewritten_input: str
    rewritten_solution: str
    test_cases: List[str]
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
                        {"role": "user", "content": prompt}
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
                                    "test_cases": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "good_program": {"type": "boolean"}
                                },
                                "required": ["rewritten_input", "rewritten_solution", "test_cases", "good_program"],
                                "additionalProperties": False
                            }
                        }
                    }
                }
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
        with open(all_files[0], "r") as f:
            try:
                response = json.load(f)
                rewritten_input = response['rewritten_input']
                if type(rewritten_input) == dict:
                    return None
                return response
            except Exception:
                return None
    
    return None

def main():
    global SAMPLE_LIMIT
    input_dataset = load_dataset(INPUT_HF_DATASET, SPLIT, split=SPLIT)
    
    # First get all unique IDs
    unique_ids = set()
    unique_rows = []
    for row in input_dataset:
        if row['id'] not in unique_ids:
            unique_ids.add(row['id'])
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
        prompts.append((get_id(row), master_prompt.replace("{input}", get_input(row)).replace("{solution}", get_solution(row))))

    print(f"Creating batch file with {len(prompts)} prompts...")
    create_batch_file(prompts)
    print(f"Created batch file at {BATCH_FILE_NAME}")

    # Submit the batch job
    print("Submitting batch job to Azure OpenAI...")
    batch_file = client.files.create(
        file=open(BATCH_FILE_NAME, "rb"),
        purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"Batch job submitted with ID: {batch_job.id}")
    print("You can check the status of your batch job using the ID above.")

if __name__ == "__main__":
    main()