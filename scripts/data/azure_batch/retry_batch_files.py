import argparse
import collections
import logging
import os
import pathlib
from typing import List

import openai

# Set up logging with file name and line number
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def retry_batch_files(batch_file_paths: List[str], output_dir: str) -> None:
    client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-04-01-preview",
    )
    print(f'{batch_file_paths=}')
    dataset_batch_ids: dict[str, List[str]] = collections.defaultdict(list)
    for batch_file_path in batch_file_paths:
        # format is timestamp_username_dataset_name_number.jsonl
        filename = pathlib.Path(batch_file_path).name
        username = filename.split("_")[1]
        dataset_name = filename.split("_")[2].split(".")[0]
        logger.info(f"Retrying batch file: {batch_file_path}, dataset: {username}/{dataset_name}.")

        batch_file = client.files.create(
            file=open(batch_file_path, "rb"), purpose="batch"
        )

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batch_id = batch_job.id
        logger.info(f"Submitted batch job with ID: {batch_id} for {dataset_name}")
        dataset_batch_ids[f"{username}/{dataset_name}"].append(batch_id)

    # Print batch IDs in the requested format
    logger.info("\nBatch job IDs by dataset:")
    for dataset_name, batch_ids in dataset_batch_ids.items():
        logger.info(f"{dataset_name}: {', '.join(batch_ids)}")
    logger.info(f"Saved batch IDs to {output_dir}/batch_ids.txt")
    all_batch_ids = [batch_id for batch_ids in dataset_batch_ids.values() for batch_id in batch_ids]
    logger.info(f"\nAll batches: {','.join(all_batch_ids)}")
    logger.info("\nYou can check the status of your batch jobs using the IDs above.")
    with open(f"{output_dir}/retry_batch_ids.txt", "a") as f:
        for dataset_name, batch_ids in dataset_batch_ids.items():
            f.write(f"{dataset_name}: {', '.join(batch_ids)}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_file_paths", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    batch_file_paths = args.batch_file_paths.split(",")
    retry_batch_files(batch_file_paths, args.output_dir)

if __name__ == "__main__":
    main()