"""
OpenAI Batch Job Creator for Code Translation

This script creates OpenAI batch jobs to translate Python coding problems into
multiple other programming languages. It sources problems from various HuggingFace
datasets and prepares them for large-scale, asynchronous processing via the
Azure OpenAI Batch API.

This is the first step in a two-part process, followed by `code_upload_batch_translate.py`
to process the results.

Features:
- Gathers and deduplicates coding problems from a list of specified HuggingFace datasets.
- For each target language, it generates prompts asking an LLM to translate the problem description, the Python solution, and its test cases.
- Uses a structured JSON output format for reliable parsing of the translated components.
- Creates a `.jsonl` batch file compatible with the OpenAI Batch API.
- Submits the batch file to Azure OpenAI to create a new batch job.
- Includes a fallback mechanism to split the batch file into smaller chunks using `split_and_submit_batch.py` if the initial submission fails.

Prerequisites:
- Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

Usage:
    The script is designed to be run to create translation batches for all languages defined in `TARGET_LANGUAGES`.

Workflow:
1.  Run this script to create and submit batch jobs for code translation.
2.  The script will output batch job IDs (or paths to files containing them).
3.  Wait for the batch jobs to complete (this can take hours for large jobs).
4.  Use `code_upload_batch_translate.py` to process the results.

Structured Output Schema:
The LLM is prompted to return a JSON object with the following keys:
- `translated_problem`: The problem description, translated.
- `translated_solution`: The solution code, translated.
- `translated_test_cases`: The test cases, translated.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import time

from datasets import Dataset, load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

MODEL = "gpt-4.1"
SAMPLE_LIMIT = None
WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)

INPUT_HF_DATASETS = [
    "saurabh5/the-algorithm-python",
    "saurabh5/llama-nemotron-rlvr",
    "saurabh5/open-code-reasoning-rlvr",
    "saurabh5/tulu-3-personas-code-rlvr",
    "saurabh5/rlvr_acecoder_filtered",
]
SPLIT = "train"

TARGET_LANGUAGES = [
    "JavaScript",
    "bash",
    "C++",
    "Go",
    "Java",
    "Rust",
    "Swift",
    "Kotlin",
    "Haskell",
    "Lean",
    "TypeScript",
]

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


# def get_input(row):
#    ds_name, ds_split, ds_index = row["dataset"], row["split"], int(row["index"])
#    if ds_name not in hf_datasets:
#        return None
#    return get_question(ds_name, ds_split, ds_index)


def get_input(row):
    return row["messages"][0]["content"]


def get_solution(row):
    """
    for message in row['messages']:
        if message['role'] == 'assistant':
            return message['content']
    return None
    """
    # return extract_python_code(row['output'])
    if "reference_solution" in row:
        return row["reference_solution"]
    elif "rewritten_solution" in row:
        return row["rewritten_solution"]
    elif "solution" in row:
        return row["solution"]
    return None


def get_test_cases(row):
    return row["ground_truth"]


def get_id(row):
    input_str = get_input(row)
    if input_str:
        return hashlib.sha256(input_str.encode("utf-8")).hexdigest()
    return None


class OpenAIStructuredOutput(BaseModel):
    translated_problem: str
    translated_solution: str
    translated_test_cases: list[str]


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
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can translate code from Python to other languages.",
                        },
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
                                    "translated_problem": {"type": "string"},
                                    "translated_solution": {"type": "string"},
                                    "translated_test_cases": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["translated_problem", "translated_solution", "translated_test_cases"],
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


def main(target_language):
    global SAMPLE_LIMIT

    input_rows = []
    for INPUT_HF_DATASET in INPUT_HF_DATASETS:
        input_dataset = load_dataset(INPUT_HF_DATASET, split=SPLIT, num_proc=open_instruct_utils.max_num_processes())
        input_rows.extend(input_dataset)

    # First get all unique IDs
    unique_ids = set()
    unique_rows = []
    for row in input_rows:
        if get_id(row) not in unique_ids:
            unique_ids.add(get_id(row))
            unique_rows.append(row)

    print(f"Found {len(unique_rows)} unique rows out of {len(input_rows)} total rows")

    # Now sample from unique rows
    random.seed(42)
    if SAMPLE_LIMIT is None:
        SAMPLE_LIMIT = len(unique_rows)
    sampled_rows = random.sample(unique_rows, min(SAMPLE_LIMIT, len(unique_rows)))

    print(f"Processing {len(sampled_rows)} unique rows")
    # create a new dataset with the rows
    new_dataset = Dataset.from_list(sampled_rows)
    new_dataset.push_to_hub("saurabh5/rlvr-code-data-python")
    master_prompt_stdio = r"""
# Instructions
Your task is to translate a Python coding problem into {target_language}. You will be given a problem description, a Python solution, and a set of test cases. Your job is to translate all three into {target_language}.

## Response Rules
- You must return only a single JSON object. No other text, explanations, or formatting.
- The JSON object must have three keys: "translated_problem", "translated_solution", and "translated_test_cases".
- Do not change the logic of the solution or the test cases. This is a translation task only.

## Problem to Translate

### Problem Description
<problem>
{input}
</problem>

### Python Solution
```python
{solution}
```

### Test Cases
```json
{test_cases}
```

## Translation Task
Translate the "Problem Description", "Python Solution", and "Test Cases" into **{target_language}**.

Output should be a JSON object with this structure:
```json
{{
    "translated_problem": "...",
    "translated_solution": "...",
    "translated_test_cases": [
        {{
            "input": "...",
            "output": "..."
        }}
    ]
}}
```
"""

    master_prompt_fn = r"""
# Instructions
Your task is to translate a Python coding problem into {target_language}. You will be given a problem description, a Python solution, and a set of test cases. Your job is to translate all three into {target_language}.

## Response Rules
- You must return only a single JSON object. No other text, explanations, or formatting.
- The JSON object must have three keys: "translated_problem", "translated_solution", and "translated_test_cases".
- Do not change the logic of the solution or the test cases. This is a translation task only.

## Problem to Translate

### Problem Description
<problem>
{input}
</problem>

### Python Solution
```python
{solution}
```

### Test Cases
{test_cases}

## Translation Task
Translate the "Problem Description", "Python Solution", and "Test Cases" into **{target_language}**.

Output should be a JSON object with this structure:
```json
{{
    "translated_problem": "...",
    "translated_solution": "...",
    "translated_test_cases": ["...", "..."]
}}
```
"""

    prompts = []
    for row in sampled_rows:
        input = get_input(row)
        solution = get_solution(row)
        test_cases = get_test_cases(row)
        if test_cases is None or solution is None or input is None:
            continue
        prompts.append(
            (
                get_id(row),
                master_prompt_fn.replace("{input}", input)
                .replace("{solution}", solution)
                .replace("{test_cases}", test_cases)
                .replace("{target_language}", target_language),
            )
        )

    print(f"Creating batch file with {len(prompts)} prompts...")
    print(f"First prompt: {prompts[0]}")
    create_batch_file(prompts)
    print(f"Created batch file at {BATCH_FILE_NAME}")

    # Submit the batch job
    try:
        print("Submitting batch job to Azure OpenAI...")
        batch_file = client.files.create(file=open(BATCH_FILE_NAME, "rb"), purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )

        print(f"Batch job submitted with ID: {batch_job.id}")
        print("You can check the status of your batch job using the ID above.")
        return batch_job.id
    except Exception as e:
        print(f"Failed to submit batch job directly: {e}")
        print("Falling back to splitting the batch file and submitting chunks.")
        # call split_and_submit_batch.py with the batch file, and name is the target language
        target_language_filename = f"{target_language}_batch_ids.json"
        script_path = os.path.join(os.path.dirname(__file__), "split_and_submit_batch.py")

        # Construct the full path where the batch IDs file will be stored.
        batch_ids_dir = os.path.join(os.getcwd(), "batch_ids")
        full_path_to_ids_file = os.path.join(batch_ids_dir, target_language_filename)

        try:
            subprocess.run(
                ["python", script_path, BATCH_FILE_NAME, "--batch_ids_file", target_language_filename], check=True
            )
            print(f"Successfully submitted chunks. Batch IDs are in {full_path_to_ids_file}")
            return full_path_to_ids_file
        except subprocess.CalledProcessError as sub_e:
            print(f"An error occurred while running split_and_submit_batch.py: {sub_e}")
            return None
        except FileNotFoundError:
            print(f"Error: The script at {script_path} was not found.")
            return None


if __name__ == "__main__":
    batch_ids = {}
    for target_language in TARGET_LANGUAGES:
        batch_ids[target_language] = main(target_language)
        break

    # with open(f"{WD}/batch_ids_translate.json", "w") as f:
    #    json.dump(batch_ids, f)
