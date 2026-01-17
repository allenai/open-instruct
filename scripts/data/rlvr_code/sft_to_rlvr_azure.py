"""
SFT to RLVR Dataset Converter using Azure OpenAI

This script converts Supervised Fine-Tuning (SFT) datasets into RLVR (Reinforcement Learning
from Verification and Reasoning) format. It processes coding problems by:
1. Transforming them into structured format with function signatures
2. Generating test cases using Azure OpenAI GPT-4.1
3. Validating solutions against test cases using a local code execution API
4. Uploading successful transformations to HuggingFace Hub

Features:
- Uses Azure OpenAI API with retry logic and rate limit handling
- S3 caching system to avoid reprocessing identical prompts
- Concurrent processing with configurable semaphore limits
- Code validation using local Docker-based execution environment
- Automatic cost tracking for API usage
- Robust error handling and logging

Prerequisites:
1. Set up Azure OpenAI API credentials:
   - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
   - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL

2. Set up AWS S3 for caching (optional but recommended):
   - Configure AWS credentials for S3 access
   - Update S3_BUCKET variable with your bucket name

3. Start the code execution API:
   ```bash
   # In the repository root directory
   docker build -t code-api -f open_instruct/code/Dockerfile .
   docker run -p 1234:1234 code-api
   ```

Usage:
    python sft_to_rlvr_azure.py

Configuration:
    Modify these variables in the script as needed:
    - INPUT_DATASET_NAME: Source HuggingFace dataset
    - OUTPUT_DATASET_NAME: Target HuggingFace dataset
    - NUM_SAMPLES: Number of samples to process
    - S3_BUCKET: S3 bucket for caching
    - READ_FROM_CACHE/WRITE_TO_CACHE: Enable/disable caching

Output:
    Creates a dataset with the following structure:
    - messages: User prompt in chat format
    - ground_truth: Test cases that pass validation
    - dataset: Dataset identifier
    - good_program: Quality flag from GPT-4.1
    - original_solution/input: Original data
    - rewritten_solution/input: Transformed data
    - reasoning-output: Original reasoning output

Example:
    ```bash
    # Set environment variables
    export AZURE_OPENAI_API_KEY="your-api-key"
    export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

    # Start code API
    docker build -t code-api -f open_instruct/code/Dockerfile .
    docker run -p 1234:1234 code-api

    # Run the script
    python sft_to_rlvr_azure.py
    ```

Note:
    This script can be expensive to run due to GPT-4.1 API costs. Monitor the cost tracking
    output and consider using a smaller NUM_SAMPLES for testing.
"""

import asyncio
import json
import os
import re  # Added for parsing retry-after
import time

import boto3  # Added for S3
import requests
from datasets import Dataset, load_dataset
from openai import AsyncAzureOpenAI, RateLimitError  # Modified to import RateLimitError
from tqdm.asyncio import tqdm_asyncio

import open_instruct.utils as open_instruct_utils

# Initialize the client with your API key
client = AsyncAzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
)

# Pricing for GPT-4.1
GPT4_1_INPUT_PRICE_PER_MILLION_TOKENS = 2.00  # USD
GPT4_1_OUTPUT_PRICE_PER_MILLION_TOKENS = 8.00  # USD

# S3 Cache Configuration
S3_BUCKET = "allennlp-saurabhs"
S3_BASE_PREFIX = "openai_responses/"  # Base prefix in S3
READ_FROM_CACHE = True
WRITE_TO_CACHE = True
INPUT_DATASET_NAME = "saurabh5/llama-nemotron-sft"
OUTPUT_DATASET_NAME = "saurabh5/llama-nemotron-rlvr"
MODEL = "gpt-4.1-standard"
NUM_SAMPLES = 50000
WD = os.getcwd()  # Retained for non-cache paths if any, though WD/open_ai_responses will no longer be used for cache
TIMESTAMP = int(time.time())
CACHE_INDEX = {}  # Global cache index: id -> s3_key
s3_client = boto3.client("s3")

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5
BACKOFF_FACTOR = 2


async def get_completion(prompt, id, semaphore):
    """Get a completion for a single prompt. Reads from/writes to S3 cache with retry logic."""
    async with semaphore:
        input_tokens = 0
        output_tokens = 0
        retries = 0
        current_backoff = INITIAL_BACKOFF_SECONDS

        while retries < MAX_RETRIES:
            try:
                if READ_FROM_CACHE:
                    openai_response_dict = find_cached_results(id)  # Expects dict or None
                    if openai_response_dict is not None:
                        return {
                            "original_prompt": prompt,
                            "openai_response": json.dumps(openai_response_dict),  # Ensure it's a JSON string
                            "success": True,
                            "input_tokens": 0,  # No new tokens consumed
                            "output_tokens": 0,  # No new tokens consumed
                        }

                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that can write code in Python."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=8192,
                )
                response_content = response.choices[0].message.content
                if response.choices[0].message.content and response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

                if response_content.startswith("```json"):
                    response_content = response_content.replace("```json", "").replace("```", "")

                if WRITE_TO_CACHE:
                    try:
                        parsed_response = json.loads(response_content)
                        # Construct S3 key using the current timestamp for new cache entries
                        s3_key = f"{S3_BASE_PREFIX}{TIMESTAMP}/openai_response_{id}.json"
                        s3_client.put_object(
                            Bucket=S3_BUCKET,
                            Key=s3_key,
                            Body=json.dumps(parsed_response, indent=2),
                            ContentType="application/json",
                        )
                        CACHE_INDEX[id] = s3_key  # Update cache index with the new S3 key
                    except json.JSONDecodeError:
                        print(f"Response for {id} is not valid JSON. Not caching to S3.")
                        pass  # response_content is still returned below if not JSON
                    except Exception as e:
                        print(f"Error caching response for {id} to S3: {e}")
                        # Decide if this should prevent returning success or just log error

                return {
                    "original_prompt": prompt,
                    "openai_response": response_content,  # Return raw content if not cached or not JSON
                    "success": True,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except RateLimitError as e:
                retries += 1
                error_message = str(e)
                print(f"Rate limit error for {id}: {error_message}. Retry {retries}/{MAX_RETRIES}.")

                retry_after_match = re.search(r"retry after (\d+) seconds", error_message, re.IGNORECASE)
                if retry_after_match:
                    wait_time = int(retry_after_match.group(1))
                    print(f"Retrying after {wait_time} seconds as suggested by API.")
                else:
                    wait_time = current_backoff
                    print(f"Retrying after {wait_time} seconds (exponential backoff).")
                    current_backoff *= BACKOFF_FACTOR

                if retries >= MAX_RETRIES:
                    print(f"Max retries reached for {id}. Giving up.")
                    return {
                        "original_prompt": prompt,
                        "openai_response": error_message,
                        "success": False,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Error in get_completion for {id}: {e}")
                return {
                    "original_prompt": prompt,
                    "openai_response": str(e),
                    "success": False,
                    "input_tokens": input_tokens,  # Could be non-zero if error happened after API call
                    "output_tokens": output_tokens,
                }
        # This part should ideally not be reached if MAX_RETRIES logic is correct within the loop
        return {
            "original_prompt": prompt,
            "openai_response": "Max retries reached, but error not propagated correctly.",
            "success": False,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


async def process_prompts(prompts):
    """Process multiple prompts concurrently."""
    # No need to os.makedirs for S3 cache, S3 handles prefixes.
    semaphore = asyncio.Semaphore(100)
    tasks = [get_completion(prompt, id, semaphore) for id, prompt in prompts]
    results = await tqdm_asyncio.gather(*tasks)
    return results


def initialize_cache_index():
    """Initializes the CACHE_INDEX by listing objects in S3."""
    global CACHE_INDEX
    print(f"Initializing cache index from S3 bucket: {S3_BUCKET}, prefix: {S3_BASE_PREFIX}...")

    temp_cache_info = {}  # id -> {s3_key, last_modified}
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_BASE_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Expected key format: S3_BASE_PREFIX{TIMESTAMP}/openai_response_{id}.json
                if key.endswith(".json") and "openai_response_" in key:
                    try:
                        # Extract ID: last part of the key after 'openai_response_', before '.json'
                        # and part of a timestamped folder
                        parts = key.split("/")
                        filename = parts[-1]
                        if filename.startswith("openai_response_"):
                            id_from_key = filename[len("openai_response_") : -len(".json")]
                            last_modified = obj["LastModified"]

                            if (
                                id_from_key not in temp_cache_info
                                or last_modified > temp_cache_info[id_from_key]["last_modified"]
                            ):
                                temp_cache_info[id_from_key] = {"s3_key": key, "last_modified": last_modified}
                    except Exception as e:
                        print(f"Could not parse S3 key {key} for ID: {e}")

        CACHE_INDEX = {id_val: data["s3_key"] for id_val, data in temp_cache_info.items()}
        print(f"S3 Cache index initialized with {len(CACHE_INDEX)} items.")
    except Exception as e:
        print(f"Error initializing S3 cache index: {e}. Ensure AWS credentials and bucket access are correct.")
        # Depending on requirements, might exit or continue with empty/partial cache.


def find_cached_results(id: str):
    """Finds cached results from S3 using the CACHE_INDEX. Returns dict or None."""
    if id in CACHE_INDEX:
        s3_key = CACHE_INDEX[id]
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            content = response["Body"].read().decode("utf-8")
            parsed_response = json.loads(content)

            # Original check for 'rewritten_input' type from previous local cache logic
            if isinstance(parsed_response.get("rewritten_input"), dict):
                print(f"Found S3 cached item for {id} ({s3_key}) but 'rewritten_input' is a dict, skipping.")
                return None
            return parsed_response  # Return the parsed dictionary
        except s3_client.exceptions.NoSuchKey:
            print(f"S3 Cache index pointed to {s3_key} for id {id}, but key not found. Removing from index.")
            del CACHE_INDEX[id]
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from S3 cached file {s3_key} for id {id}. Skipping.")
            return None
        except Exception as e:
            print(f"Error reading S3 cached file {s3_key} for id {id}: {e}")
            return None
    return None


# Example usage
async def main():
    if READ_FROM_CACHE:  # Only initialize if cache is enabled
        initialize_cache_index()

    dataset = load_dataset(INPUT_DATASET_NAME, split="train", num_proc=open_instruct_utils.max_num_processes())
    dataset = dataset.filter(lambda x: x["solution"] != "NO SOLUTION" and x["generator"] == "DeepSeek-R1")
    print(f"Samples after filtering: {len(dataset)}")

    dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"Processing {len(dataset)} rows")

    master_prompt = r"""\
# Instructions
Your task is to transform coding problems into a structured dataset format with function-based solutions and test cases.

## Response Rules
- Return only a JSON object with no additional text
- The JSON must include: rewritten_input, rewritten_solution, test_cases, and good_program
- Set good_program to False if the solution is incorrect or the problem is unsuitable

## Transformation Process
1. Convert the input problem to specify a function signature
2. Rewrite the solution to use function parameters instead of input()
3. Create test cases as executable assert statements
4. Package everything in the required JSON format

## Input Requirements
- Keep the rewritten_input similar in length to the original
- Clearly specify the function name and parameters
- Update any examples to use the new function signature

## Test Case Requirements
- Extract test cases from the input when available
- Add new test cases to cover edge cases
- Format as executable Python assert statements
- Do not include comments in test cases

Here is the file input:
<INPUT>
{input}
</INPUT>

Here is a reference solution:
```python
{solution}
```

Output should be a JSON object with this structure:
{
    "rewritten_input": "...",
    "rewritten_solution": "...",
    "test_cases": ["...", "..."],
    "good_program": true/false
}"""
    prompts = []
    for row in dataset:
        prompts.append(
            (
                row["id"],
                master_prompt.replace("{input}", row["input"][0]["content"]).replace("{solution}", row["solution"]),
            )
        )

    print(f"Processing {len(prompts)} prompts asynchronously...")
    results = await process_prompts(prompts)

    total_input_tokens = 0
    total_output_tokens = 0

    url = "http://localhost:1234/test_program"
    new_results = []
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        total_input_tokens += result.get("input_tokens", 0)
        total_output_tokens += result.get("output_tokens", 0)
        try:
            # Result openai_response is already a string, potentially JSON string
            openai_response_json_str = result["openai_response"]
            openai_response_data = json.loads(openai_response_json_str)  # Parse it to dict

            test_cases = openai_response_data["test_cases"]
            rewritten_solution = openai_response_data["rewritten_solution"]
            rewritten_input = openai_response_data["rewritten_input"]

            payload = {"program": rewritten_solution, "tests": test_cases, "max_execution_time": 2.0}
            response = requests.post(url, json=payload)
            response_json = response.json()
            if response.ok and sum(response_json["results"]) >= 0.85 * len(test_cases):
                passed_test_cases = [test_cases[j] for j in range(len(test_cases)) if response_json["results"][j] == 1]
                new_results.append(
                    {
                        "messages": [{"role": "user", "content": rewritten_input}],
                        "ground_truth": passed_test_cases,
                        "dataset": "code",
                        "good_program": openai_response_data["good_program"],
                        "original_solution": dataset[i]["solution"],
                        "original_input": dataset[i]["input"],
                        "reasoning-output": dataset[i]["output"],
                        "rewritten_solution": rewritten_solution,
                        "rewritten_input": rewritten_input,
                        "id": dataset[i]["id"],
                    }
                )
            else:
                print("Not adding. ")
                print(f"OpenAI response: {openai_response_data}")
                print(f"Execution results: {response_json}")
        except Exception as e:
            print(f"Error parsing or processing result for item {i}: {e}")
            print(f"Original OpenAI response string: {result.get('openai_response', 'N/A')}")
            import traceback

            print(f"Traceback:\n{traceback.format_exc()}")

    print(f"After filtering, {len(new_results)} results out of {len(results)} remain")
    if new_results:  # Only proceed if there are results to push
        dataset_to_push = Dataset.from_list(new_results)
        dataset_to_push.push_to_hub(OUTPUT_DATASET_NAME)
    else:
        print("No results to push to hub.")

    # Calculate and print total cost
    cost_input = (total_input_tokens / 1_000_000) * GPT4_1_INPUT_PRICE_PER_MILLION_TOKENS
    cost_output = (total_output_tokens / 1_000_000) * GPT4_1_OUTPUT_PRICE_PER_MILLION_TOKENS
    total_cost = cost_input + cost_output

    print("\n--- Token Usage and Cost ---")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Estimated Cost for Input Tokens: ${cost_input:.4f}")
    print(f"Estimated Cost for Output Tokens: ${cost_output:.4f}")
    print(f"Total Estimated Cost: ${total_cost:.4f}")

    print("all done")


if __name__ == "__main__":
    asyncio.run(main())
