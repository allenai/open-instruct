"""
OpenAI Batch Job Results Processor for Code Translation

This script retrieves the results of completed OpenAI batch jobs that were created to
translate Python coding problems into other programming languages. It collects the
translated outputs and uploads them as new datasets to the HuggingFace Hub.

This script is the second step in a two-part workflow, used after a batch job has been
created by a script like `code_create_batch_translate.py`.

Features:
- Retrieves results from multiple completed OpenAI batch jobs.
- Reads batch job IDs from a central JSON file (`batch_ids_translate.json`), which maps target languages to their corresponding batch ID files.
- For each target language, it loads the batch job IDs and fetches the results.
- Parses the JSON output from the batch job, which contains the translated problem, solution, and test cases.
- Constructs new HuggingFace Datasets from the retrieved results.
- Uploads the datasets to the HuggingFace Hub, creating a separate repository for each target language.

Prerequisites:
- Azure OpenAI API credentials must be set as environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).
- The batch jobs for translation must be completed.
- A `batch_ids_translate.json` file must exist, mapping target languages to the files containing their batch job IDs.

Usage:
    python code_upload_batch_translate.py

Workflow:
1. A batch creation script (e.g., `code_create_batch_translate.py`) is run to submit translation jobs to OpenAI.
2. Wait for the batch jobs to complete.
3. Run this script to retrieve the results and upload them to the HuggingFace Hub.
"""

import hashlib
import json
import os

from datasets import Dataset, load_dataset
from openai import AzureOpenAI
from pydantic import BaseModel

import open_instruct.utils as open_instruct_utils

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

OUTPUT_HF_DATASET_PREFIX = "saurabh5/rlvr-code-data"
SPLIT = "train"
BATCH_ID_FILEPATH = "batch_ids_translate.json"

TARGET_LANGUAGS = [
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


# needed for open-code-reasoning-2
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


# def get_input(row):
#    ds_name, ds_split, ds_index = row["dataset"], row["split"], int(row["index"])
#    if ds_name not in hf_datasets:
#        return None
#    return get_question(ds_name, ds_split, ds_index)


def get_input(row):
    return row["input"][0]["content"]


def get_id(row):
    input_str = get_input(row)
    if input_str:
        id = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        return id
    return None


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


def process_batch_results(batch_ids: list[str], target_language: str):
    """Process the batch results and upload to Hugging Face."""
    batch_results = []
    for batch_id in batch_ids:
        # every batch should be complete
        if not check_batch_status(batch_id):
            print(f"Batch {batch_id} is not complete yet")
            return

        batch_results.extend(get_batch_results(batch_id))
        if not batch_results:
            print("No results found in batch")
            continue

    new_results = []
    for result in batch_results:
        try:
            translated_test_cases = result["translated_test_cases"]

            new_result = {
                **result,
                "messages": [{"role": "user", "content": result["translated_problem"]}],
                "translated_problem": result["translated_problem"],
                "translated_solution": result["translated_solution"],
                "ground_truth": json.dumps(translated_test_cases),
                "target_language": target_language,
            }
            new_results.append(new_result)
        except Exception as e:
            print(f"Error processing result: {e}")
            print(f"Result: {result}")

    print(
        f"About to upload {len(new_results)} results to Hugging Face out of {len(batch_results)}. Do you want to upload?"
    )

    # Upload to Hugging Face
    if new_results:
        dataset = Dataset.from_list(new_results)
        if target_language == "C++":
            target_language = "cpp"
        dataset.push_to_hub(f"{OUTPUT_HF_DATASET_PREFIX}-{target_language}")
        print(f"Uploaded {len(new_results)} examples to {OUTPUT_HF_DATASET_PREFIX}-{target_language}")
    else:
        print("No valid results to upload")


if __name__ == "__main__":
    batch_ids = json.load(open(BATCH_ID_FILEPATH))
    for language, batch_id_filepath in batch_ids.items():
        batch_ids = json.load(open(batch_id_filepath))
        process_batch_results(batch_ids, language)
