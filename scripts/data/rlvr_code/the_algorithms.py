"""
TheAlgorithms Python Dataset Generator

This script processes Python algorithm implementations from TheAlgorithms/Python repository
and converts them into a structured dataset format suitable for code generation training.
It extracts algorithms, creates human-like prompts, generates test cases, and validates
the reference solutions.

Features:
- Recursively processes all Python files in TheAlgorithms repository
- Uses OpenAI GPT-4o to generate human-like prompts and test cases
- Validates solutions using local code execution API
- Filters out low-quality or unsuitable programs
- Uploads results to HuggingFace Hub as a dataset

Prerequisites:
1. Set up OpenAI API credentials:
   - OPENAI_API_KEY: Your OpenAI API key

2. Clone TheAlgorithms repository:
   ```bash
   git clone https://github.com/TheAlgorithms/Python.git
   ```

3. Start the code execution API:
   ```bash
   # In the repository root directory
   docker build -t code-api -f open_instruct/code/Dockerfile .
   docker run -p 1234:1234 code-api
   ```

Usage:
    ```bash
    # Clone the algorithms repository
    git clone https://github.com/TheAlgorithms/Python.git

    # Set your OpenAI API key
    export OPENAI_API_KEY="your-openai-api-key"

    # Start the code execution API
    docker build -t code-api -f open_instruct/code/Dockerfile .
    docker run -p 1234:1234 code-api

    # Run the script
    python the_algorithms.py
    ```

Output:
    Creates a dataset with the following structure:
    - messages: User prompt asking to implement the algorithm
    - ground_truth: Test cases as executable assert statements
    - dataset: Dataset identifier ("vwxyzjn/the-algorithm-python")
    - reference_solution: The original algorithm implementation
    - good_program: Quality flag indicating if the program is suitable

Process:
1. Scans all .py files in the Python directory (excluding __init__.py)
2. Sends each file to OpenAI to generate:
   - A human-like prompt asking for the algorithm implementation
   - Test cases extracted from the original code
   - A reference solution
3. Validates the reference solution against test cases
4. Keeps only programs where all test cases pass
5. Uploads the filtered dataset to HuggingFace Hub

Note:
    This script processes a large number of files and makes many API calls to OpenAI.
    Consider the API costs and rate limits when running on the full repository.
"""

import asyncio
import json
import os

import requests
from datasets import Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Initialize the client with your API key
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Set your API key as an environment variable
)


async def get_completion(prompt):
    """Get a completion for a single prompt."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",  # You can change the model as needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can write code in Python."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8192,
        )
        return {"original_prompt": prompt, "openai_response": response.choices[0].message.content, "success": True}
    except Exception as e:
        return {"original_prompt": prompt, "openai_response": str(e), "success": False}


async def process_prompts(prompts):
    """Process multiple prompts concurrently."""
    tasks = [get_completion(prompt) for prompt in prompts]
    results = await tqdm_asyncio.gather(*tasks)
    return results


# Example usage
async def main():
    # List of files to process
    # files = [
    #     "Python/backtracking/all_combinations.py",
    #     "Python/backtracking/combination_sum.py",
    # ]
    # read all the `.py` files in the `Python` directory
    # Get full paths to avoid FileNotFoundError
    # base_dir = "Python/backtracking"
    base_dir = "Python"
    file_contents = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path) as f:
                        content = f.read()
                        file_contents.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Processing {len(file_contents)} files")
    breakpoint()

    # Create prompts to process
    master_prompt = r"""\
I am creating a dataset of prompt and test cases. Can you convert the following file to
1) a prompt asking the user to implement a function and 2) a list of test cases extracted from the file,
and 3) the reference solution to the function.

Here is the file:
{file_content}

The prompt should be more human-like questions: relatively concise.
The test cases should be a python list of executable assert code.

You should return in json format like below. I should be able to parse it directly using json.loads, so
in your response, please do not include any other text than the json. Also do not inlcude the ```json tags.
Feel free to use nice formats like bullet points in the prompt.

If you don't think the file is a good candidate for this dataset, you can set the "good_program" to False.

{
    "prompt": ...,
    "test_cases": [...],
    "reference_solution": ...,
    "good_program": ...,
}"""
    prompts = []
    for file_content in file_contents:
        prompts.append(master_prompt.replace("{file_content}", file_content))

    print(f"Processing {len(prompts)} prompts asynchronously...")
    results = await process_prompts(prompts)

    # Filter: make sure reference solution is correct and solves the test cases
    url = "http://localhost:1234/test_program"
    new_results = []
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        try:
            openai_response = json.loads(result["openai_response"])
            reference_solution = openai_response["reference_solution"]
            test_cases = openai_response["test_cases"]

            # Test data
            payload = {"program": reference_solution, "tests": test_cases, "max_execution_time": 2.0}
            # Send POST request
            response = requests.post(url, json=payload)
            response_json = response.json()
            if sum(response_json["results"]) == len(test_cases):
                new_results.append(
                    {
                        "messages": [{"role": "user", "content": openai_response["prompt"]}],
                        "ground_truth": test_cases,
                        "dataset": "vwxyzjn/the-algorithm-python",
                        "reference_solution": reference_solution,
                        "good_program": openai_response["good_program"],
                    }
                )
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {result['openai_response']}")

    print(f"Filtered {len(new_results)} results out of {len(results)}")

    dataset = Dataset.from_list(new_results)
    dataset.push_to_hub("vwxyzjn/the-algorithm-python")

    print("all done")


if __name__ == "__main__":
    asyncio.run(main())
