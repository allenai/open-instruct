#!/usr/bin/env python3
"""
check_batch.py ― quick status checker for Azure OpenAI Batch API jobs.

Usage:
    export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
    export AZURE_OPENAI_KEY="<your-key>"
    # optional override (defaults to the latest preview that includes Batch)
    # export AZURE_OPENAI_API_VERSION="2024-07-01-preview"

    python check_batch.py <batch_id> [--watch]

With --watch it polls every 15 s until the job leaves the “in_progress/validating/finalizing”
states and then exits with a non-zero code on failure.
"""
from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser, Namespace

import requests

TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        sys.stderr.write(f"missing required env var {name}\n")
        sys.exit(1)


def build_url(batch_id: str) -> str:
    endpoint = env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")
    return f"{endpoint}/openai/batches/{batch_id}?api-version={api_version}"


def fetch(batch_id: str) -> dict:
    r = requests.get(
        build_url(batch_id),
        headers={"api-key": env("AZURE_OPENAI_API_KEY")},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def print_status(obj: dict) -> None:
    status = obj["status"]
    counts = obj.get("request_counts", {})
    line = (
        f"{status}"
        f" | {counts.get('completed', 0)}/{counts.get('total', '?')} done"
        f" | {counts.get('failed', 0)}/{counts.get('total', '?')} failed"
        f" | batch_id={obj['id']}"
    )
    print(line, flush=True)


def cli() -> Namespace:
    p = ArgumentParser(description="Check Azure OpenAI batch job status")
    p.add_argument("batch_id", help="batch_… identifier returned at creation")
    p.add_argument("--watch", action="store_true", help="poll until terminal state")
    p.add_argument("--interval", type=int, default=15, help="seconds between polls")
    return p.parse_args()


def main() -> None:
    args = cli()
    while True:
        job = fetch(args.batch_id)
        print(f'{job=}')
        print_status(job)
        if not args.watch or job["status"] in TERMINAL_STATES:
            sys.exit(0 if job["status"] == "completed" else 1)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
