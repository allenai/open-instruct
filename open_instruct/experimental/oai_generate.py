"""
This script is used to generate responses from a model using the OpenAI API.

It supports 1) chat completion async for quick debugging and 2) batch for large scale generation.

Usage:

## Setup Environment Variables
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="..."
```

## Async chat completion
```bash
python open_instruct/experimental/oai_generate.py --input_file open_instruct/experimental/test.jsonl --output_file test_oai_output.jsonl
```
```
Using `AZURE_OPENAI_API_KEY` from environment variable
Using `AZURE_OPENAI_ENDPOINT` from environment variable
Input token count: 19
Estimated input token cost (based on GPT-4o): $0.000038
100%|██████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.38s/it]
Output saved to test_oai_output.jsonl
Output token count: 619
Estimated output token cost (based on GPT-4o): $0.004952
```

## Batch

This will run for quite a while. You should only run this in a CPU-only machine and come back in a few hours.
```bash
python open_instruct/experimental/oai_generate.py --input_file open_instruct/experimental/test.jsonl --output_file test_oai_output.jsonl --batch
```
```
{
  "id": "file-522e90e573a342f4a91f956d5eb273e8",
  "bytes": 800,
  "created_at": 1744306253,
  "filename": "test.jsonl",
  "object": "file",
  "purpose": "batch",
  "status": "processed",
  "status_details": null
}
{
  "id": "batch_b369f218-0d56-4922-af13-88aa97916d21",
  "completion_window": "24h",
  "created_at": 1744306254,
  "endpoint": "/chat/completions",
  "input_file_id": "file-522e90e573a342f4a91f956d5eb273e8",
  "object": "batch",
  "status": "validating",
  "cancelled_at": null,
  "cancelling_at": null,
  "completed_at": null,
  "error_file_id": "",
  "errors": null,
  "expired_at": null,
  "expires_at": 1744392654,
  "failed_at": null,
  "finalizing_at": null,
  "in_progress_at": null,
  "metadata": null,
  "output_file_id": "",
  "request_counts": {
    "completed": 0,
    "failed": 0,
    "total": 0
  }
}
2025-04-10 13:30:59.888211 Batch Id: batch_b369f218-0d56-4922-af13-88aa97916d21,  Status: validating
...
2025-04-10 13:35:10.416338 Batch Id: batch_b369f218-0d56-4922-af13-88aa97916d21,  Status: in_progress
...
2025-04-10 13:37:23.851718 Batch Id: batch_b369f218-0d56-4922-af13-88aa97916d21,  Status: finalizing
...
2025-04-10 13:40:43.572143 Batch Id: batch_b369f218-0d56-4922-af13-88aa97916d21,  Status: completed
Output saved to test_oai_output.jsonl
```
"""

import asyncio
import datetime
import json
import os
import time
from dataclasses import dataclass

import tiktoken
from openai import AsyncAzureOpenAI, AzureOpenAI
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio
from transformers import HfArgumentParser

console = Console()


@dataclass
class Args:
    input_file: str
    output_file: str
    batch: bool = False
    batch_check_interval: int = 10
    api_key: str | None = None
    api_version: str = "2024-12-01-preview"
    azure_endpoint: str | None = None

    def __post_init__(self):
        if self.api_key is None and "AZURE_OPENAI_API_KEY" in os.environ:
            print("Using `AZURE_OPENAI_API_KEY` from environment variable")
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if self.api_key is None and "OPENAI_API_KEY" in os.environ:
            print("Using `OPENAI_API_KEY` from environment variable")
            self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "API key is required"

        if self.azure_endpoint is None and "AZURE_OPENAI_ENDPOINT" in os.environ:
            print("Using `AZURE_OPENAI_ENDPOINT` from environment variable")
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert self.azure_endpoint is not None, "Azure endpoint is required"


# @vwxyzjn: price per 1M input tokens
# https://platform.openai.com/docs/pricing#latest-models
# as of 2025-04-16
PRICE_PER_1M_INPUT_TOKENS_GPT_4O = 2.00
PRICE_PER_1M_OUTPUT_TOKENS_GPT_4O = 8.00


def main(args: Args):
    enc = tiktoken.encoding_for_model("gpt-4o")
    input_price_per_token = PRICE_PER_1M_INPUT_TOKENS_GPT_4O / 1000000
    output_price_per_token = PRICE_PER_1M_OUTPUT_TOKENS_GPT_4O / 1000000
    if args.batch:
        input_price_per_token /= 2
        output_price_per_token /= 2
    input_token_count = 0
    with open(args.input_file) as infile:
        for line in infile:
            data = json.loads(line)
            input_token_count += len(enc.encode(data["body"]["messages"][-1]["content"]))
    console.print(f"[bold green]Input token count: {input_token_count}[/bold green]")
    estimated_input_token_cost = input_token_count * input_price_per_token
    console.print(
        f"[bold green]Estimated input token cost (based on GPT-4o): ${estimated_input_token_cost:.6f}[/bold green]"
    )

    if args.batch:
        client = AzureOpenAI(api_key=args.api_key, api_version=args.api_version, azure_endpoint=args.azure_endpoint)
        with open(args.input_file, "rb") as input_file:
            file = client.files.create(file=input_file, purpose="batch")

        print(file.model_dump_json(indent=2))
        file_id = file.id

        # Submit a batch job with the file
        batch_response = client.batches.create(
            input_file_id=file_id, endpoint="/chat/completions", completion_window="24h"
        )

        # Save batch ID for later use
        batch_id = batch_response.id
        print(batch_response.model_dump_json(indent=2))

        # Track batch job progress
        status = "validating"
        while status not in ("completed", "failed", "canceled"):
            time.sleep(args.batch_check_interval)
            batch_response = client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response.status == "failed":
            for error in batch_response.errors.data:
                print(f"Error code {error.code} Message {error.message}")

        # Retrieve batch job output file
        output_file_id = batch_response.output_file_id
        if not output_file_id:
            output_file_id = batch_response.error_file_id
        if output_file_id:
            file_response = client.files.content(output_file_id)
            raw_responses = file_response.text.strip().split("\n")
            # Save the output to x30_output.jsonl
            with open(args.output_file, "w") as outfile:
                for raw_response in raw_responses:
                    outfile.write(raw_response + "\n")
            console.print(f"[bold green]Output saved to {args.output_file}[/bold green]")
    else:
        client = AsyncAzureOpenAI(
            api_key=args.api_key, api_version=args.api_version, azure_endpoint=args.azure_endpoint
        )

        async def async_main():
            tasks = []

            async def create_task(data):
                body = data["body"]
                # @vwxyzjn: hack because of the azure weird stuff
                # gpt-4o is required for batch, but gpt-4o-standard is required for streaming
                body["model"] = body["model"].replace("gpt-4o", "gpt-4o-standard")
                response = await client.chat.completions.create(**data["body"])
                return response

            with open(args.input_file) as infile:
                for line in infile:
                    data = json.loads(line)
                    tasks.append(create_task(data))
            responses = await tqdm_asyncio.gather(*tasks)
            # Save the output to x30_output.jsonl
            with open(args.output_file, "w") as outfile:
                for idx, response in enumerate(responses):
                    outfile.write(
                        json.dumps({"custom_id": f"task-{idx}", "response": {"body": response.model_dump()}}) + "\n"
                    )
            console.print(f"[bold green]Output saved to {args.output_file}[/bold green]")

        asyncio.run(async_main())

    # estimate output token cost
    output_token_count = 0
    with open(args.output_file) as infile:
        for line in infile:
            data = json.loads(line)
            output_token_count += len(enc.encode(data["response"]["body"]["choices"][0]["message"]["content"]))
    console.print(f"[bold green]Output token count: {output_token_count}[/bold green]")
    estimated_output_token_cost = output_token_count * output_price_per_token
    console.print(
        f"[bold green]Estimated output token cost (based on GPT-4o): ${estimated_output_token_cost:.6f}[/bold green]"
    )


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
