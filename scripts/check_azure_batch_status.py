#!/usr/bin/env python3
"""
check_batch.py – enhanced Azure OpenAI Batch status checker.

Features
--------
* One-liner status output identical to the original.
* Prints per-batch *errors* (high-level codes/messages).
* Can **download** the row-level *error_file_id* and
  **peek** at the first N failed rows.

Environment
-----------
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<key>"
# optional
export AZURE_OPENAI_API_VERSION="2024-07-01-preview"

Examples
--------
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

TERMINAL = {"completed", "failed", "cancelled", "expired"}
CHUNK = 1 << 20  # 1 MiB stream buffer


# ---------- helpers -----------------------------------------------------------
def env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        sys.stderr.write(f"missing required env var {name}\n")
        sys.exit(1)


def api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")


def build_url(batch_id: str) -> str:
    endpoint = env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    return f"{endpoint}/openai/batches/{batch_id}?api-version={api_version()}"


def fetch(batch_id: str) -> dict:
    r = requests.get(
        build_url(batch_id),
        headers={"api-key": env("AZURE_OPENAI_API_KEY")},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fmt_errors(errs: list[dict] | None) -> str:
    if not errs:
        return "-"
    return ";".join(
        f"{e.get('code') or e.get('error_code')}:{e.get('message')}" for e in errs
    )


def download_file(file_id: str, dest: pathlib.Path) -> None:
    endpoint = env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    url = f"{endpoint}/openai/files/{file_id}/content?api-version={api_version()}"
    with requests.get(url, headers={"api-key": env("AZURE_OPENAI_API_KEY")}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)


def print_status(job: dict) -> None:
    c = job.get("request_counts", {})
    line = (
        f"{job['status']}"
        f" | ok {c.get('completed', 0)}/{c.get('total', '?')}"
        f" | fail {c.get('failed', 0)}"
        f" | errors {fmt_errors(job.get('errors'))}"
        f" | id={job['id']}"
    )
    print(line, flush=True)


# ---------- CLI ---------------------------------------------------------------
def cli() -> Namespace:
    p = ArgumentParser(description="Check Azure OpenAI batch job status")
    p.add_argument("batch_id")
    p.add_argument("--watch", action="store_true", help="poll until terminal state")
    p.add_argument("--interval", type=int, default=15, help="seconds between polls")
    p.add_argument("--peek-errors", type=int, metavar="N", help="print first N rows from error_file_id")
    p.add_argument("--save-errors", metavar="PATH", help="download full error_file_id to PATH")
    return p.parse_args()


# ---------- main loop ---------------------------------------------------------
def main() -> None:
    args = cli()

    while True:
        job = fetch(args.batch_id)
        print_status(job)

        if job["status"] in TERMINAL:
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
