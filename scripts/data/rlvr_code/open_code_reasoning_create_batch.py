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
from openai import AzureOpenAI, OpenAI
from datasets import load_dataset
from pydantic import BaseModel
from typing import List
from collections import Counter
import random
# Initialize the client with your API key
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
)


MODEL = "gpt-4.1"
NUM_SAMPLES = 20000
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

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
                "url": "/chat/completions",
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
    ocr_ds_split_0 = load_dataset("nvidia/OpenCodeReasoning", 'split_0', split="split_0")
    
    # First get all unique IDs
    unique_ids = set()
    unique_rows = []
    for row in ocr_ds_split_0:
        if row['id'] not in unique_ids:
            unique_ids.add(row['id'])
            unique_rows.append(row)
    
    print(f"Found {len(unique_rows)} unique rows out of {len(ocr_ds_split_0)} total rows")
    
    # Now sample from unique rows
    random.seed(42)
    sampled_rows = random.sample(unique_rows, min(NUM_SAMPLES, len(unique_rows)))
    
    print(f"Processing {len(sampled_rows)} unique rows")

    master_prompt = r"""\
I am creating a dataset of input, test cases, and a solution. 
I want you to rewrite the solution so that the test cases can call one function with arguements.
The solution should not use input() to get the input. It should use the arguments passed to the function.
Also, rewrite the input so that a person or an LLM reading it can write a program with the correct signature.
The input should ask for a program of a certain signature. The solution should be a program of that signature.
The test cases should be a python list of executable assert code which calls the program in the solution.
Extract the test cases from the input and add new test cases in addition to extracting the existing ones.

The test cases should be a python list of executable assert code. 
The reference solution aims to be correct, so the rewritten solution should be close to the reference solution.
The rewritten input should also be close to the original input, but it should specify the signature of the program and the examples should use that program signature instead of the original input method.
The test cases should be as comprehensive as possible, covering all edge cases.

You should return in json format like below. I should be able to parse it directly using json.loads, so
in your response, please do not include any other text than the json.

If you don't think the solution is correct, or the problem is not a good candidate for this dataset, you can set the "good_program" to False.
The tests cases should be a list of assert statements. Do not put any comments in the test cases, as they will break the json parsing.
The test cases should call the rewritten solution that you provide.
The rewritten_input should be a string similar to the original <INPUT>, but it should specify the signature of the program and the examples should use that program signature instead of the original input method.
The rewritten_input should not be a json object. It should be about the same length as the original <INPUT>, and it should mention the name of the function and the parameters it takes. If the original input includes examples, the rewritten input should include examples that use the new function signature.

Here is the file input:
<INPUT>
{input}
</INPUT>

Here is a reference solution:
```python
{solution}
```

Output should be a json object like this:
{
    "rewritten_input": ...,
    "rewritten_solution": ...,
    "test_cases": [...],
    "good_program": ...,
}"""

    prompts = []
    for row in sampled_rows:
        prompts.append((row['id'], master_prompt.replace("{input}", row['input']).replace("{solution}", row['solution'])))

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
        endpoint="/chat/completions",
        completion_window="24h"
    )

    print(f"Batch job submitted with ID: {batch_job.id}")
    print("You can check the status of your batch job using the ID above.")

if __name__ == "__main__":
    main()