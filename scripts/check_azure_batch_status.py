#!/usr/bin/env python3
"""Check the status of an Azure OpenAI batch job.

Usage:

# just once:
./check_batch.py batch_abc

# watch until done, print two error rows, save full error file:
./check_batch.py batch_abc --watch --peek-errors 2 --save-errors bad.jsonl
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from argparse import ArgumentParser, Namespace

import requests


def download_file(file_id: str, dest: pathlib.Path) -> None:
    """Download a file from Azure OpenAI API to the specified destination."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version=2024-07-01-preview"
    
    response = requests.get(
        url,
        headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
        timeout=120
    )
    response.raise_for_status()
    
    with dest.open("wb") as f:
        f.write(response.content)


def print_status(job: dict) -> None:
    c = job.get("request_counts", {})
    errors = job.get("errors", [])
    error_str = "-" if not errors else ";".join(
        f"{e.get('code') or e.get('error_code')}:{e.get('message')}" for e in errors
    )
    line = (
        f"{job['status']}"
        f" | ok {c.get('completed', 0)}/{c.get('total', '?')}"
        f" | fail {c.get('failed', 0)}"
        f" | errors {error_str}"
        f" | id={job['id']}"
    )
    print(line, flush=True)


def cli() -> Namespace:
    p = ArgumentParser(description="Check Azure OpenAI batch job status")
    p.add_argument("batch_id")
    p.add_argument("--watch", action="store_true", help="poll until terminal state")
    p.add_argument("--interval", type=int, default=15, help="seconds between polls")
    p.add_argument("--peek-errors", type=int, metavar="N", help="print first N rows from error_file_id")
    p.add_argument("--save-errors", metavar="PATH", help="download full error_file_id to PATH")
    return p.parse_args()


def main() -> None:
    args = cli()

    while True:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        url = f"{endpoint}/openai/batches/{args.batch_id}?api-version=2024-07-01-preview"
        r = requests.get(
            url,
            headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
            timeout=30,
        )
        r.raise_for_status()
        job = r.json()
        
        print_status(job)

        if job["status"] in {"completed", "failed", "cancelled", "expired"}:
            if (args.peek_errors or args.save_errors) and job.get("error_file_id"):
                dest = pathlib.Path(args.save_errors or ".errors.jsonl")
                download_file(job["error_file_id"], dest)

                if args.peek_errors:
                    with dest.open() as f:
                        for _ in range(args.peek_errors):
                            try:
                                print(json.loads(next(f)))
                            except StopIteration:
                                break
            sys.exit(0 if job["status"] == "completed" else 1)

        if not args.watch:
            sys.exit(1)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
