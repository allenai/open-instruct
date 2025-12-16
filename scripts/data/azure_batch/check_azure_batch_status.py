#!/usr/bin/env python3
"""Check the status of an Azure OpenAI batch job.

Usage:

# just once:
./check_batch.py batch_abc
./check_batch.py batch_abc,batch_def,batch_ghi

# watch until done:
./check_batch.py batch_abc --watch
./check_batch.py batch_abc,batch_def,batch_ghi --watch
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from argparse import ArgumentParser, Namespace

import requests

# Constants
POLL_INTERVAL = 15  # seconds between status checks


def download_file(file_id: str, dest: pathlib.Path) -> None:
    """Download a file from Azure OpenAI API to the specified destination."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version=2024-07-01-preview"

    response = requests.get(url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=120)
    response.raise_for_status()

    with dest.open("wb") as f:
        f.write(response.content)


def print_status(job: dict) -> None:
    c = job.get("request_counts", {})
    errors = job.get("errors", [])
    print(f"{errors=}")
    error_str = (
        "-" if not errors else ";".join(f"{e.get('code') or e.get('error_code')}:{e.get('message')}" for e in errors)
    )
    line = (
        f"{job['status']}"
        f" | ok {c.get('completed', 0)}/{c.get('total', '?')}"
        f" | fail {c.get('failed', 0)}"
        f" | errors {error_str}"
        f" | id={job['id']}"
    )
    print(line, flush=True)


def load_jsonl(file_path: pathlib.Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with file_path.open() as f:
        return [json.loads(line) for line in f]


def show_errors_with_requests(job: dict) -> None:
    """Download and display errors alongside their corresponding requests."""
    if not job.get("error_file_id"):
        print("No error file available")
        return

    # Create temporary files for downloads
    error_file = pathlib.Path(".errors.jsonl")
    result_file = pathlib.Path(".results.jsonl")

    try:
        # Download error file
        download_file(job["error_file_id"], error_file)
        errors = load_jsonl(error_file)

        # If we have a result file, download it too
        results_by_id = {}
        if job.get("result_file_id"):
            download_file(job["result_file_id"], result_file)
            results = load_jsonl(result_file)
            results_by_id = {r.get("id"): r for r in results}

        print("\nErrors with corresponding requests (showing first 10):")
        print("-" * 80)

        for error in errors[:10]:  # Only show first 10 errors
            request_id = error.get("id")
            result = results_by_id.get(request_id, {})

            print(f"\nError ID: {request_id}")
            print(f"Error: {error.get('error', {}).get('message', 'Unknown error')}")

            # Show the original request data from the error file
            if "request" in error:
                print(f"Original Request: {json.dumps(error['request'], indent=2)}")
            # Fall back to result file data if available
            elif result:
                print(f"Request: {json.dumps(result.get('request', {}), indent=2)}")
            print("-" * 80)

        if len(errors) > 10:
            print(f"\n... and {len(errors) - 10} more errors not shown")

    finally:
        # Clean up temporary files
        error_file.unlink(missing_ok=True)
        result_file.unlink(missing_ok=True)


def cli() -> Namespace:
    p = ArgumentParser(description="Check Azure OpenAI batch job status")
    p.add_argument("batch_ids", help="Single batch ID or comma-separated list of batch IDs")
    p.add_argument("--watch", action="store_true", help="poll until terminal state")
    return p.parse_args()


def check_batch_status(batch_id: str) -> tuple[bool, bool]:
    """Check status of a single batch job.

    Returns:
        tuple[bool, bool]: (is_terminal_state, is_success)
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/batches/{batch_id}?api-version=2024-07-01-preview"
    r = requests.get(url, headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]}, timeout=30)
    r.raise_for_status()
    job = r.json()

    print(f"\nBatch ID: {batch_id}")
    print_status(job)

    if job["status"] in {"completed", "failed", "cancelled", "expired"}:
        if job.get("error_file_id"):
            show_errors_with_requests(job)
        return True, job["status"] == "completed"

    return False, False


def main() -> None:
    args = cli()
    batch_ids = [id.strip() for id in args.batch_ids.split(",")]

    while True:
        all_terminal = True
        all_success = True

        for batch_id in batch_ids:
            is_terminal, is_success = check_batch_status(batch_id)
            all_terminal = all_terminal and is_terminal
            all_success = all_success and is_success

        if all_terminal:
            sys.exit(0 if all_success else 1)

        if not args.watch:
            sys.exit(1)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
