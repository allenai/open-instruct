"""
OpenAI Batch Results Processor for Code Difficulty Grading

This script retrieves the results from completed OpenAI batch jobs that were created to
grade the difficulty of coding problems. It merges these difficulty scores back into the
original datasets and uploads the updated datasets to the HuggingFace Hub.

This script is the second step in a two-part workflow, designed to be used after a batch
job has been created by `grade_difficulty.py`.

Features:
- Retrieves difficulty grading results from completed OpenAI batch jobs.
- Reads a mapping of datasets to their corresponding batch job IDs from `batches_difficulty.json`.
- For each dataset, it loads the original data from Hugging Face.
- It fetches the batch results, which contain difficulty scores and explanations for each problem.
- Merges the difficulty information into the original dataset based on a matching problem ID.
- Pushes the updated dataset, now including `difficulty` and `difficulty_explanation` columns, back to its original repository on the HuggingFace Hub.

Prerequisites:
- Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).
- The batch jobs for difficulty grading must be in "completed" status.
- A `batches_difficulty.json` file must exist, mapping Hugging Face dataset names to their corresponding OpenAI batch job IDs.

Usage:
    python code_upload_batch_difficulty.py

Workflow:
1.  Run `grade_difficulty.py` to submit batch jobs that ask an LLM to rate the difficulty of problems in one or more datasets.
2.  Wait for the batch jobs to complete.
3.  Run this script to retrieve the difficulty scores, merge them into the source datasets, and upload the updated datasets to the HuggingFace Hub.
"""

import json
import os

from datasets import Dataset, load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel, ConfigDict

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

BATCH_ID_FILEPATH = "batches_difficulty.json"


def get_id(row):
    return row["id"]


def extract_id_from_custom_id(custom_id: str) -> str:
    # get rid of the timestamp
    if "_" in custom_id:
        return "_".join(custom_id.split("_")[1:])
    return custom_id


class OpenAIStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str
    difficulty: int


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


def process_batch_results(dataset: str, batch_id: str):
    """Process the batch results and upload to Hugging Face."""
    batch_results = get_batch_results(batch_id)
    if not batch_results:
        print("No results found in batch")
        return

    ds = load_dataset(dataset, split="train", num_proc=open_instruct_utils.max_num_processes())
    print("building id to row")
    id_to_row = {get_id(row): row for row in ds}
    new_results = []
    for result in batch_results:
        try:
            id = result["id"]
            if id not in id_to_row:
                print(f"ID {id} not found in dataset")
                continue

            row = id_to_row[id]
            new_row = {**row}
            new_row["difficulty"] = result["difficulty"]
            new_row["difficulty_explanation"] = result["explanation"]
            new_results.append(new_row)
        except Exception as e:
            print(f"Error processing result: {e}")
            print(f"Result: {result}")

    print(
        f"About to upload {len(new_results)} results to Hugging Face out of {len(batch_results)}. Do you want to upload?"
    )

    # Upload to Hugging Face
    if new_results:
        upload_dataset = Dataset.from_list(new_results)
        repo_id = dataset
        upload_dataset.push_to_hub(repo_id)
        print(f"Uploaded {len(new_results)} examples to {repo_id}")
    else:
        print("No valid results to upload")


if __name__ == "__main__":
    batch_ids = json.load(open(BATCH_ID_FILEPATH))
    for dataset, batch_id in batch_ids.items():
        process_batch_results(dataset, batch_id)
