"""This script is used to regenerate completions for a dataset using a specific OpenAI model.

Usage:

Cd into the directory of this file and run:
```
python regenerate_dataset_completions.py
```

"""
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel


def get_input(row: dict[str, str]) -> str:
    return row['input']

def get_solution(row: dict[str, str]) -> str:
    """
    for message in row['messages']:
        if message['role'] == 'assistant':
            return message['content']
    return None
    """
    return row['solution']

def get_id(row: dict[str, str]) -> str:
    return row['id']

class OpenAIStructuredOutput(BaseModel):
    rewritten_input: str
    rewritten_solution: str
    test_cases: List[str]
    good_program: bool

@dataclass
class PromptData:
    id: str
    prompt: str

def create_batch_file(prompts: List[PromptData], batch_file_name: str, model: str, timestamp: int) -> None:
    """Create a batch file in the format required by Azure OpenAI Batch API."""
    with open(batch_file_name, "w") as f:
        for prompt in prompts:
            # Format each request according to batch API requirements
            batch_request = {
                "custom_id": f"{timestamp}_{prompt.id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that can write code in Python."},
                        {"role": "user", "content": prompt.prompt}
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

    with open(all_files[0], "r") as f:
        try:
            response = json.load(f)
            rewritten_input = response['rewritten_input']
            if isinstance(rewritten_input, dict):
                return None
            return response
        except Exception:
            return None

def main(sample_limit: int | None = None,
         input_dataset_name: str = "nvidia/OpenCodeReasoning",
         split: str = "split_0",
         model: str = "gpt-4.1",
         current_dir: str | None = None) -> None:
    
    if current_dir is None:
        current_dir = os.getcwd()
    
    timestamp = int(time.time())
    batch_file_name = f"{current_dir}/batch_files/{timestamp}.jsonl"
    
    # Make sure that the batch files directory exists.
    os.makedirs(f"{current_dir}/batch_files", exist_ok=True)

    input_dataset = load_dataset(input_dataset_name, split)
    
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
    sample_limit = len(unique_rows) if sample_limit is None else sample_limit
    sampled_rows = random.sample(unique_rows, min(sample_limit, len(unique_rows)))
    
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

    prompts: List[PromptData] = []
    for row in sampled_rows:
        prompts.append(PromptData(id=get_id(row), prompt=master_prompt.replace("{input}", get_input(row)).replace("{solution}", get_solution(row))))

    print(f"Creating batch file with {len(prompts)} prompts...")
    create_batch_file(prompts, batch_file_name, model, timestamp)
    print(f"Created batch file at {batch_file_name}")

    quit()


    # Initialize the client with your API key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Submit the batch job
    print("Submitting batch job to Azure OpenAI...")
    batch_file = client.files.create(
        file=open(batch_file_name, "rb"),
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
