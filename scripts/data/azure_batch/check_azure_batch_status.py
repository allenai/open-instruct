#!/usr/bin/env python3
"""Check the status of an OpenAI batch job.

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
from typing import List

from openai import OpenAI

# Constants
POLL_INTERVAL = 15  # seconds between status checks

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def download_file(file_id: str, dest: pathlib.Path) -> None:
    """Download a file from OpenAI API to the specified destination."""
    response = client.files.content(file_id)
    with dest.open("wb") as f:
        f.write(response.content)


def print_status(job) -> None:
    """Prints the status of a batch job."""
    c = job.request_counts
    total_requests = c.total if c else "?"
    completed_requests = c.completed if c else 0
    failed_requests = c.failed if c else 0

    error_str = "-"
    if job.errors and job.errors.data:
        errors = job.errors.data
        error_str = ";".join(f"{e.code}:{e.message}" for e in errors)

    line = (
        f"{job.status}"
        f" | ok {completed_requests}/{total_requests}"
        f" | fail {failed_requests}"
        f" | errors {error_str}"
        f" | id={job.id}"
    )
    print(line, flush=True)


def load_jsonl(file_path: pathlib.Path) -> List[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with file_path.open() as f:
        return [json.loads(line) for line in f]


def show_errors(job) -> None:
    """Download and display errors."""
    if not job.error_file_id:
        print("No error file available")
        return

    # Create temporary files for downloads
    error_file = pathlib.Path(".errors.jsonl")

    try:
        # Download error file
        download_file(job.error_file_id, error_file)
        errors = load_jsonl(error_file)

        print("\nErrors (showing first 10):")
        print("-" * 80)

        for error in errors[:10]:  # Only show first 10 errors
            custom_id = error.get("custom_id", "N/A")
            error_details = error.get("error", {})
            print(f"\nCustom ID: {custom_id}")
            print(f"Error: {json.dumps(error_details, indent=2)}")
            print("-" * 80)

        if len(errors) > 10:
            print(f"\n... and {len(errors) - 10} more errors not shown")

    finally:
        # Clean up temporary files
        error_file.unlink(missing_ok=True)


def cli() -> Namespace:
    p = ArgumentParser(description="Check OpenAI batch job status")
    p.add_argument("batch_ids", help="Single batch ID or comma-separated list of batch IDs")
    p.add_argument("--watch", action="store_true", help="poll until terminal state")
    return p.parse_args()


def check_batch_status(batch_id: str) -> tuple[bool, bool]:
    """Check status of a single batch job.
    
    Returns:
        tuple[bool, bool]: (is_terminal_state, is_success)
    """
    job = client.batches.retrieve(batch_id)

    print(f"\nBatch ID: {batch_id}")
    print_status(job)

    if job.status in {"completed", "failed", "cancelled", "expired"}:
        if job.error_file_id:
            show_errors(job)
        return True, job.status == "completed"

    return False, False


def main() -> None:
    args = cli()
    batch_ids = [id.strip() for id in args.batch_ids.split(",")]

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

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
