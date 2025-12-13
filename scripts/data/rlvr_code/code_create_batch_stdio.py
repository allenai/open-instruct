"""
OpenAI Batch Job Creator for stdio Code Transformation

This script creates OpenAI batch jobs to transform function-based Python coding
problems into a format that uses standard input/output (stdio). It reads problems
from a Hugging Face dataset, generates prompts asking an LLM to perform the
transformation, and then submits a batch job to the Azure OpenAI API.

This is the first step in a two-part process, followed by `code_upload_batch_stdio.py`
to process the results.

Features:
- Loads a coding dataset from the HuggingFace Hub.
- Deduplicates problems to avoid reprocessing.
- Generates detailed prompts that instruct an LLM to:
  - Rewrite the problem description to specify interaction via stdio.
  - Rewrite the solution to use stdio for input and output (e.g., `input()` and `print()`).
  - Generate a list of test cases, each with an `input` and `output` field.
- Uses a structured JSON output format for reliable parsing of the results.
- Creates a `.jsonl` batch file and submits it to the Azure OpenAI API.

Prerequisites:
- Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

Usage:
    python code_create_batch_stdio.py

Workflow:
1.  Run this script to create and submit a batch job for stdio transformation.
2.  The script will output a batch job ID.
3.  Wait for the batch job to complete.
4.  Use `code_upload_batch_stdio.py` with the job ID to validate and upload the results.

Structured Output Schema:
The LLM is prompted to return a JSON object with the following keys:
- `rewritten_input`: The problem description, rewritten for stdio.
- `rewritten_solution`: The solution code, rewritten for stdio.
- `test_cases`: A list of dictionaries, where each is a test case with `input` and `output` strings.
- `good_program`: A boolean flag indicating if the transformation was successful.
"""

import json
import os
import random
import re
import time

from datasets import load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel, ConfigDict

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

MODEL = "o3"
SAMPLE_LIMIT = 200_000
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

INPUT_HF_DATASET = "nvidia/OpenCodeReasoning-2"
SPLIT = "python"


def extract_python_code(model_output: str) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return model_output

    # Return the last match, stripped of whitespace
    return matches[-1].strip()


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
    dataset = row["dataset"]
    split = row["split"]
    index = row["index"]
    return get_question(dataset, split, index)


def get_solution(row):
    return row["solution"]


def get_id(row):
    return row["question_id"]


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str
    output: str


class OpenAIStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rewritten_input: str
    rewritten_solution: str
    test_cases: list[TestCase]
    good_program: bool


def create_batch_file(prompts):
    """Create a batch file in the format required by Azure OpenAI Batch API."""
    # Generate JSON schema from Pydantic models.
    # Using model_json_schema is more robust than crafting the JSON by hand.
    schema = OpenAIStructuredOutput.model_json_schema(ref_template="#/definitions/{model}")
    if "$defs" in schema:
        # pydantic v2 uses $defs, but older specs might expect definitions
        schema["definitions"] = schema.pop("$defs")

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
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can write code in Python and use stdio.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "OpenAIStructuredOutput", "strict": True, "schema": schema},
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
    input_dataset = load_dataset(INPUT_HF_DATASET, split=SPLIT, num_proc=open_instruct_utils.max_num_processes())

    # First get all unique IDs
    unique_ids = set()
    unique_rows = []
    for row in input_dataset:
        if get_id(row) not in unique_ids:
            unique_ids.add(get_id(row))
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
Your task is to transform coding problems into a structured dataset format with stdio-based solutions and test cases.

## Response Rules
- Return only a JSON object with no additional text
- The JSON must include: rewritten_input, rewritten_solution, test_cases, and good_program
- Set good_program to False if the solution is incorrect or the problem is unsuitable

## Transformation Process
1. Rewrite the solution to use stdio, such as print() and input(). Make minimal changes to the solution. Changes might not be needed.
2. Create test cases. These test cases should be a list of dictionaries with the following keys:
    - input: The input to the program. If there are multiple inputs for a single test case, separate them with a newline.
    - output: The expected output of the program. If there are multiple outputs for a single test case, separate them with a newline.
3. Package everything in the required JSON format

## Rewritten Input Requirements
- The rewritten_input should be a complete problem description.
- If the original problem is phrased in terms of function inputs and outputs (e.g., "write a function `solve(n)`..."), rewrite it to specify interaction via standard input and standard output.
- Clearly describe the format for standard input and expected standard output.
- If the original problem already describes interaction via standard input and output, you can keep it as is.
- Keep the rewritten_input similar in length to the original.

## Test Case Requirements
- Extract test cases from the input when available
- Add new test cases to cover edge cases
- You can use test cases from the input when available, but also you should add new test cases to cover edge cases and so they are hidden from the user
- Do not include comments in test cases

## Example
Here is an example of a program that uses stdio. The input is the following problem:

<INPUT>
Let p (i) be the i-th prime number from the smallest. For example, 7 is the fourth prime number from the smallest, 2, 3, 5, 7, so p (4) = 7.\n\nGiven n, the sum of p (i) from i = 1 to n s\n\ns = p (1) + p (2) + .... + p (n)\n\nCreate a program that outputs. For example, when n = 9, s = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100.\nInput\n\nGiven multiple datasets. Each dataset is given the integer n (n ≤ 10000). When n is 0, it is the last input. The number of datasets does not exceed 50.\n\nOutput\n\nFor n in each dataset, print s on one line.\n\nExample\n\nInput\n\n2\n9\n0\n\n\nOutput\n\n5\n100
</INPUT>

Here is a reference solution for this example:
```python
def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5)+1, 2):
        if num % i == 0:
            return False
    return True

def generate_primes_up_to_nth(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def main():
    import sys
    inputs = sys.stdin.read().split()
    for line in inputs:
        n = int(line)
        if n == 0:
            break
        primes = generate_primes_up_to_nth(n)
        print(sum(primes))

if __name__ == "__main__":
    main()
```

And so you would return the following JSON object:
{{
    "rewritten_input": "Let p (i) be the i-th prime number from the smallest. For example, 7 is the fourth prime number from the smallest, 2, 3, 5, 7, so p (4) = 7.\n\nGiven n, the sum of p (i) from i = 1 to n s\n\ns = p (1) + p (2) + .... + p (n)\n\nCreate a program that outputs. For example, when n = 9, s = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100.\nInput\n\nGiven multiple datasets. Each dataset is given the integer n (n ≤ 10000). When n is 0, it is the last input. The number of datasets does not exceed 50.\n\nOutput\n\nFor n in each dataset, print s on one line.\n\nExample\n\nInput\n\n2\n9\n0\n\n\nOutput\n\n5\n100",
    "rewritten_solution": "def is_prime(num):\n    if num <= 1:\n        return False\n    if num == 2:\n        return True\n    if num % 2 == 0:\n        return False\n    for i in range(3, int(num**0.5)+1, 2):\n        if num % i == 0:\n            return False\n    return True\n\ndef generate_primes_up_to_nth(n):\n    primes = []\n    num = 2\n    while len(primes) < n:\n        if is_prime(num):\n            primes.append(num)\n        num += 1\n    return primes\n\ndef main():\n    import sys\n    inputs = sys.stdin.read().split()\n    for line in inputs:\n        n = int(line)\n        if n == 0:\n            break\n        primes = generate_primes_up_to_nth(n)\n        print(sum(primes))\n\nif __name__ == \"__main__\":\n    main()",
    "test_cases": [{{"input": "2\n9\n0\n", "output": "5\n100\n"}},
  {{"input": "1\n0\n", "output": "2\n"}},
  {{"input": "0\n", "output": ""}},
  {{"input": "10\n0\n", "output": "129\n"}},
  {{"input": "5\n3\n2\n0\n", "output": "28\n10\n5\n"}},
  {{"input": "10000\n0\n", "output": "496165411\n"}},
    "good_program": true
}}

Now that you've seen an example, time for the real problem.
"""

    prompts = []
    for row in sampled_rows:
        input_text = get_input(row)
        if input_text is None:
            continue
        solution_text = get_solution(row)

        # Using an f-string for performance
        prompt = f"""{master_prompt}## Problem
Here is the problem input:
<INPUT>
{input_text}
</INPUT>

Here is a reference solution:
```python
{solution_text}
```

Output should be a JSON object with this structure:
{{
    "rewritten_input": "...",
    "rewritten_solution": "...",
    "test_cases": [{{ "input": "...", "output": "..." }}, {{ "input": "...", "output": "..." }}],
    "good_program": true/false
}}
"""
        prompts.append((get_id(row), prompt))

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
