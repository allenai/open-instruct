#!/usr/bin/env python3
"""List all Azure OpenAI batch jobs.

This script lists all batch jobs associated with the provided Azure OpenAI API key.
It reads the endpoint and API key from environment variables.

Usage:
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export AZURE_OPENAI_API_KEY="your_api_key"
./list_azure_batches.py
"""

import os
import requests
import json
import datetime

def print_batch_summary(job: dict) -> None:
    """Prints a one-line summary of a batch job."""
    c = job.get("request_counts", {})
    
    created_at_ts = job.get("created_at")
    if created_at_ts:
        try:
            created_at_str = datetime.datetime.fromtimestamp(created_at_ts).strftime('%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            created_at_str = str(created_at_ts)
    else:
        created_at_str = "?"
        
    line = (
        f"ID: {job.get('id', 'N/A')}"
        f" | Status: {job.get('status', 'N/A'):<12}"
        f" | Created: {created_at_str}"
        f" | Completed: {c.get('completed', 0)}/{c.get('total', '?')}"
        f" | Failed: {c.get('failed', 0)}"
    )
    print(line)

def main() -> None:
    """Lists all batch jobs."""
    try:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
    except KeyError as e:
        print(f"Error: Environment variable {e} not set. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
        exit(1)

    # Using a recent preview version. You might need to adjust this.
    api_version = "2024-07-01-preview"
    limit = 100  # Number of jobs to fetch per request, max is 100
    base_url = f"{endpoint}/openai/batches?api-version={api_version}&limit={limit}"
    
    print("Fetching batch jobs...")
    
    all_jobs = []
    url = base_url
    
    try:
        while True:
            r = requests.get(
                url,
                headers={"api-key": api_key},
                timeout=30,
            )
            r.raise_for_status()
            response_json = r.json()
            
            jobs_page = response_json.get("data", [])
            all_jobs.extend(jobs_page)

            if response_json.get("has_more"):
                last_id = response_json.get("last_id")
                if not last_id:
                    print("Error: 'has_more' is true but 'last_id' is missing.")
                    break
                url = f"{base_url}&after={last_id}"
            else:
                break
        
        if not all_jobs:
            print("No batch jobs found.")
            return

        # The API returns most recent first, let's reverse to show oldest first.
        all_jobs.reverse()

        print("-" * 80)
        for job in all_jobs:
            print_batch_summary(job)
        print("-" * 80)
        print(f"Total jobs: {len(all_jobs)}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while communicating with the Azure API: {e}")
        if e.response is not None:
            try:
                print("Error details:", e.response.json())
            except json.JSONDecodeError:
                print("Could not decode error response. Status code:", e.response.status_code)
                print("Response content:", e.response.text)
        exit(1)


if __name__ == "__main__":
    main() 