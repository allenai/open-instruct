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

## Batch
```bash
python open_instruct/experimental/oai_generate.py --input_file open_instruct/experimental/test.jsonl --output_file test_oai_output.jsonl --batch
```
"""


import asyncio
from dataclasses import dataclass
import json
import os
from typing import Optional
from openai import AzureOpenAI, AsyncAzureOpenAI
import time
import datetime 
from tqdm.asyncio import tqdm_asyncio
from transformers import HfArgumentParser

@dataclass
class Args:
    input_file: str
    output_file: str
    batch: bool = False
    batch_check_interval: int = 10
    api_key: Optional[str] = None
    api_version: str = "2024-12-01-preview"
    azure_endpoint: Optional[str] = None

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


def main(args: Args):
    if args.batch:
        client = AzureOpenAI(
            api_key=args.api_key,  
            api_version=args.api_version,
            azure_endpoint = args.azure_endpoint
        )
        # Upload a file with a purpose of "batch"
        file = client.files.create(
            file=open(args.input_file, "rb"), 
            purpose="batch"
        )

        print(file.model_dump_json(indent=2))
        file_id = file.id

        # Submit a batch job with the file
        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
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
            raw_responses = file_response.text.strip().split('\n')
            # Save the output to x30_output.jsonl
            with open(args.output_file, "w") as outfile:
                for raw_response in raw_responses:
                    outfile.write(raw_response + '\n')
            print(f"Output saved to {args.output_file}")
    else:
        client = AsyncAzureOpenAI(
            api_key=args.api_key,  
            api_version=args.api_version,
            azure_endpoint = args.azure_endpoint
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

            with open(args.input_file, "r") as infile:
                for line in infile:
                    data = json.loads(line)
                    tasks.append(create_task(data))
            responses = await tqdm_asyncio.gather(*tasks)
            # Save the output to x30_output.jsonl
            with open(args.output_file, "w") as outfile:
                for idx, response in enumerate(responses):
                    outfile.write(
                        json.dumps({
                            "custom_id": f"task_{idx}",
                            "response": {
                                "body": response.model_dump(),
                            },
                        }) + '\n')
            print(f"Output saved to {args.output_file}")

        asyncio.run(async_main())

if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
