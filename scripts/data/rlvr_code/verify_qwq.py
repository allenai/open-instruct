"""
run source configs/beaker_configs/code_api_setup.sh before running this script.
Resource intensive, so run as a batch job, not in a session. E.g:

python mason.py \
    --cluster ai2/saturn \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --description "filtering correct python qwq generations" \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 0 -- source configs/beaker_configs/code_api_setup.sh \\&\\& python scripts/data/rlvr_code/verify_qwq.py
"""

import asyncio
import hashlib
import os
import re
import time

import aiohttp
from datasets import Dataset, load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import open_instruct.utils as open_instruct_utils
from open_instruct import logger_utils

# Set up logging
logger = logger_utils.setup_logger(__name__)

CONCURRENCY_LIMIT = 256
SOURCE_DATASET = "saurabh5/correct-python-sft-187k-x16-thoughts"
DESTINATION_DATASET = "saurabh5/correct-python-sft-187k-x16-thoughts-filtered"


def load_dataset_with_retries(dataset_name, split="train", max_retries=5, initial_backoff=2):
    """Loads a dataset from Hugging Face with retries and exponential backoff."""
    retries = 0
    backoff = initial_backoff
    while retries < max_retries:
        try:
            return load_dataset(dataset_name, split=split, num_proc=open_instruct_utils.max_num_processes())
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit exceeded for {dataset_name}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2  # Exponentially increase the wait time
            else:
                # Re-raise other types of errors immediately
                raise e
    raise Exception(f"Failed to load dataset {dataset_name} after {max_retries} retries.")


def get_id(row):
    return hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()


def extract_python_code(model_output: str) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return model_output

    # Return the last match, stripped of whitespace
    return matches[-1].strip()


async def verify_and_process_row(session, row, id_to_row, base_url, dataset_to_url, semaphore):
    async with semaphore:
        id = row["question_id"]
        if id not in id_to_row:
            return "skipped", None

        og_row = id_to_row[id]
        python_code = extract_python_code(row["messages"][1]["content"])
        source = row["source"]
        full_url = f"{base_url}{dataset_to_url[source]}"
        payload = {"program": python_code, "tests": og_row["ground_truth"], "max_execution_time": 6.0}

        try:
            async with session.post(full_url, json=payload, timeout=10) as response:
                response.raise_for_status()
                result = await response.json()

                # Calculate pass rate from the results
                test_results = result.get("results", [])
                if not test_results:
                    pass_rate = 0.0
                else:
                    # In the API, a result of 1 means pass.
                    successful_tests = sum(1 for r in test_results if r == 1)
                    pass_rate = successful_tests / len(test_results)

                if pass_rate == 1.0:
                    return "passed", row
                else:
                    return "failed", None
        except Exception as e:
            logger.error(f"Request failed for question_id {id}: {e}")
            return "failed", None


async def main():
    dataset_to_url = {
        "saurabh5/open-code-reasoning-rlvr-stdio": "/test_program_stdio",
        "saurabh5/rlvr_acecoder_filtered": "/test_program",
        "saurabh5/llama-nemotron-rlvr-code-stdio": "/test_program_stdio",
        "saurabh5/the-algorithm-python": "/test_program",
    }
    dataset_map = {}
    id_to_row = {}
    for dataset_name, url in dataset_to_url.items():
        dataset_map[dataset_name] = load_dataset_with_retries(dataset_name)
        for row in tqdm(dataset_map[dataset_name], desc=f"Processing {dataset_name}"):
            id_to_row[get_id(row)] = row

    base_url = os.getenv("CODE_API_URL")
    logger.info(f"Using code API URL: {base_url}")

    source_ds = load_dataset_with_retries(SOURCE_DATASET)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks = [
            verify_and_process_row(session, row, id_to_row, base_url, dataset_to_url, semaphore) for row in source_ds
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Verifying generated code")

    filtered_rows = []
    skipped_rows = 0
    failed_rows = 0

    for status, row in results:
        if status == "passed":
            filtered_rows.append(row)
        elif status == "skipped":
            skipped_rows += 1
        elif status == "failed":
            failed_rows += 1

    logger.info(f"Skipped {skipped_rows} rows")
    logger.info(f"Failed {failed_rows} rows")
    logger.info(f"Passed {len(filtered_rows)} rows, uploading to {DESTINATION_DATASET}")

    filtered_ds = Dataset.from_list(filtered_rows)
    filtered_ds.push_to_hub(DESTINATION_DATASET)


if __name__ == "__main__":
    asyncio.run(main())
