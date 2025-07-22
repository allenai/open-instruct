import os
import time
from openai import AzureOpenAI, OpenAI
import json
import argparse

# Number of lines to include in each smaller batch file
CHUNK_SIZE = 18_000

def main():
    """
    Splits a large OpenAI batch file into smaller chunks and submits each as a separate batch job.

    This script is a utility to handle very large batch files that might exceed OpenAI's limits
    or fail during submission. It improves the reliability of submitting large-scale data processing
    jobs by breaking them into manageable pieces.

    Features:
    - Reads a large batch file (`.jsonl`).
    - Splits the file into smaller chunks of a configurable size (`CHUNK_SIZE`).
    - Submits each chunk as an independent batch job to the Azure OpenAI API.
    - Implements a retry mechanism for submission failures.
    - Logs the job IDs of all successfully submitted chunks into a JSON file for later tracking.

    Usage:
        python split_and_submit_batch.py <source_batch_file> [--batch_ids_file <output_filename>]

    Arguments:
        - `source_batch_file`: The path to the large `.jsonl` batch file to be split.
        - `--batch_ids_file` (optional): The name of the JSON file where the submitted batch job IDs will be stored. If not provided, it defaults to a name based on the source file.
    """
    parser = argparse.ArgumentParser(description="Split a large batch file and submit for processing.")
    parser.add_argument("source_batch_file", type=str, help="The full path to the large batch file.")
    parser.add_argument("--batch_ids_file", type=str, help="The name for the file to store submitted batch IDs.")
    args = parser.parse_args()
    """
    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-12-01-preview"
    )
    """
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if not os.path.exists(args.source_batch_file):
        print(f"Error: Source batch file not found at {args.source_batch_file}")
        return

    file_basename = os.path.basename(args.source_batch_file).split('.')[0]
    batch_dir = os.path.dirname(args.source_batch_file)

    batch_ids_dir = os.path.join(os.getcwd(), "batch_ids")
    os.makedirs(batch_ids_dir, exist_ok=True)

    if args.batch_ids_file:
        ids_log_file = os.path.join(batch_ids_dir, args.batch_ids_file)
    else:
        ids_log_file = os.path.join(batch_ids_dir, f"{file_basename}_batch_ids.json")

    submitted_job_ids = []

    with open(args.source_batch_file, 'r') as f_in:
        chunk_num = 1
        lines = []
        for line in f_in:
            lines.append(line)
            if len(lines) == CHUNK_SIZE:
                job_id = submit_chunk(client, lines, chunk_num, file_basename, batch_dir)
                if job_id:
                    submitted_job_ids.append(job_id)
                lines = []
                chunk_num += 1
        
        if lines: # Submit the last partial chunk
            job_id = submit_chunk(client, lines, chunk_num, file_basename, batch_dir)
            if job_id:
                submitted_job_ids.append(job_id)

    if submitted_job_ids:
        with open(ids_log_file, 'w') as f_out:
            json.dump(submitted_job_ids, f_out)
        print(f"\nAll processing complete. Submitted batch job IDs are saved in: {ids_log_file}")


def submit_chunk(client, lines, chunk_num, base_name, batch_dir):
    """
    Writes a chunk to a file and submits it as a batch job.
    """
    chunk_filename = os.path.join(batch_dir, f"{base_name}_chunk_{chunk_num}.jsonl")
    print(f"\nProcessing chunk {chunk_num} with {len(lines)} lines...")

    with open(chunk_filename, 'w') as f_out:
        f_out.writelines(lines)
    
    print(f"Created chunk file: {chunk_filename}")

    for attempt in range(3):
        try:
            print("Uploading batch file...")
            with open(chunk_filename, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")
            
            print(f"File uploaded successfully. File ID: {batch_file.id}")

            print("Creating batch job...")
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(f"--> Batch job for chunk {chunk_num} submitted successfully. Job ID: {batch_job.id}")
            return batch_job.id # Success, exit the function
        except Exception as e:
            print(f"Attempt {attempt + 1} for chunk {chunk_num} failed: {e}")
            if attempt < 2:
                print("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"--> Failed to submit chunk {chunk_num} after 3 attempts.")
                break # Failed after retries
    return None


if __name__ == "__main__":
    main() 