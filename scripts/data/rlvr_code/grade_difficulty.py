"""
This script uses the Azure OpenAI API to grade the difficulty of coding problems from various Hugging Face datasets.

It works by:
1.  Defining a list of source Hugging Face datasets containing coding problems.
2.  For each dataset, it generates prompts that ask a large language model to assign a difficulty score (0-10) and an explanation for each problem.
3.  These prompts are formatted into a batch file compliant with the Azure OpenAI Batch API.
4.  The script submits this batch file as a batch job to Azure OpenAI.
5.  The script stores the batch job IDs for each dataset in a JSON file (`batches_difficulty.json`) so the results can be retrieved later.

The expected output from the language model for each problem is a JSON object containing `explanation` and `difficulty`.
"""

import json
import os
import time

from datasets import load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel, ConfigDict

import open_instruct.utils as open_instruct_utils

datasets = [
    "saurabh5/open-code-reasoning-rlvr",
    "saurabh5/tulu-3-personas-code-rlvr",
    "saurabh5/rlvr_acecoder_filtered",
    "saurabh5/the-algorithm-python",
    "saurabh5/llama-nemotron-rlvr",
    "saurabh5/open-code-reasoning-rlvr-stdio",
]
MODEL = "o3"


client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

WD = os.getcwd()
TIMESTAMP = int(time.time())
BATCH_FILE_NAME = f"{WD}/batch_files/{TIMESTAMP}.jsonl"
os.makedirs(f"{WD}/batch_files", exist_ok=True)


prompt = r"""
You are a helpful assistant that grades the difficulty of a code problem.

You will be given a code problem and a solution to the problem.

You will need to grade the difficulty of the problem based on the both the solution and the problem.

The difficulty is a number between 0 and 10, where 0 is the easiest and 10 is the hardest.
The explanation should explain why you gave the difficulty you did.

Structure your response with the following JSON format:

{{
    "explanation": <explanation>,
    "difficulty": <difficulty>
}}

Here is the code problem:

<problem>
{problem}
</problem>

Here is the solution:
<solution>
{solution}
</solution>
"""


class OpenAIStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str
    difficulty: int


def get_input(row):
    return row["messages"][0]["content"]


def get_solution(row):
    if "rewritten_solution" in row:
        return row["rewritten_solution"]
    elif "reference_solution" in row:
        return row["reference_solution"]
    else:
        return row["solution"]


def get_id(row):
    return row["id"]


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
                            "content": "You are a helpful assistant that can understand code and grade the difficulty of a Python code problem.",
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


def grade_difficulty(dataset):
    ds = load_dataset(dataset, split="train", num_proc=open_instruct_utils.max_num_processes())
    prompts = []
    for row in ds:
        problem = get_input(row)
        solution = get_solution(row)
        prompts.append((get_id(row), prompt.format(problem=problem, solution=solution)))
    create_batch_file(prompts)
    print(f"Created batch file at {BATCH_FILE_NAME}")

    print("Submitting batch job to Azure OpenAI...")
    batch_file = client.files.create(file=open(BATCH_FILE_NAME, "rb"), purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
    )

    print(f"Batch job submitted with ID: {batch_job.id}")

    return batch_job.id


if __name__ == "__main__":
    batches = {}
    for dataset in datasets:
        print(f"processing {dataset}")
        batches[dataset] = grade_difficulty(dataset)

    # save batches to json
    with open(f"{WD}/batches_difficulty.json", "w") as f:
        json.dump(batches, f)
