"""
Usage:

In the root directory, run:
```
docker build -t code-api -f open_instruct/code/Dockerfile .
docker run -p 1234:1234 code-api
```

Cd into the directory of this file and run:
```
python open_code_reasoning.py
```
"""
import asyncio
import json
import os
import time
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import requests
from datasets import Dataset, load_dataset, concatenate_datasets

# Initialize the client with your API key
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Set your API key as an environment variable
)

USE_CACHE = False
DATASET_NAME = "saurabh5/open-code-reasoning-rlvr"
MODEL = "gpt-4o"
NUM_SAMPLES = 1000
WD = os.getcwd()
TIMESTAMP = int(time.time())
    

async def get_completion(prompt, id):
    """Get a completion for a single prompt."""
    try:
        if USE_CACHE:
            openai_response = find_cached_results(id)
            if openai_response is not None:
                print(f"Found cached results for {id}")
                return {
                    "original_prompt": prompt,
                    "openai_response": json.dumps(openai_response),
                    "success": True
                }


        response = await client.chat.completions.create(
            model=MODEL,  # You can change the model as needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can write code in Python."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
        )
        response = response.choices[0].message.content
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")
        

        # only cache valid json
        try:
            parsed_response = json.loads(response)
            with open(f"{WD}/open_ai_responses/{TIMESTAMP}/openai_response_{id}.json", "w") as f:
                json.dump(parsed_response, f, indent=2)
        except json.JSONDecodeError:
            pass

        return {
            "original_prompt": prompt,
            "openai_response": response,
            "success": True
        }
    except Exception as e:
        return {
            "original_prompt": prompt,
            "openai_response": str(e),
            "success": False
        }

async def process_prompts(prompts):
    """Process multiple prompts concurrently."""
    os.makedirs(f"{WD}/open_ai_responses/{TIMESTAMP}", exist_ok=True)
    tasks = [get_completion(prompt, id) for id, prompt in prompts]
    results = await tqdm_asyncio.gather(*tasks)
    return results

def find_cached_results(id: str):
    response_dir = f"{WD}/open_ai_responses/"
    
    # Recursively walk the directory
    all_files = []
    for root, _, files in os.walk(response_dir):
        for file in files:
            if file.endswith(f"openai_response_{id}.json"):
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    
    # Sort files by modification time, newest first
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return the contents of the first matching file
    if all_files:
        with open(all_files[0], "r") as f:
            try:
                response = json.load(f)
                rewritten_input = response['rewritten_input']
                if type(rewritten_input) == dict:
                    return None
                return response
            except Exception:
                return None
    
    return None

# Example usage
async def main():

    # Read data from hf dataset: nvidia/OpenCodeReasoning
    dataset = load_dataset("nvidia/OpenCodeReasoning", 'split_0', split="split_0")
    # Shuffle and take random 1000 samples
    dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))

    
    print(f"Processing {len(dataset)} rows")
    breakpoint()

    # Create prompts to process
    master_prompt = r"""\
I am creating a dataset of input, test cases, and a solution. 
I want you to rewrite the solution so that the test cases can call one function with arguements.
The solution should not use input() to get the input, it should use the arguments passed to the function.
Also, rewrite the input so that a person or an LLM reading it can write a program with the correct signature.
The input should ask for a program of a certain signature. The solution should be a program of that signature.
The test cases should be a python list of executable assert code which calls the program in the solution.
Extract the test cases from the input and feel free to add new test cases in addition to extracting the existing ones.

Here is the file input:
<INPUT>
{input}
</INPUT>

Here is a reference solution:
```python
{solution}
```

The test cases should be a python list of executable assert code. 
The reference solution aims to be correct, so the rewritten solution should be close to the reference solution.
The rewrtiten input should also be close to the original input, but it should specific the signature of the program and the examples should use that program signature instead of the original input method.
The test cases should be as comprehensive as possible, covering all edge cases.

You should return in json format like below. I should be able to parse it directly using json.loads, so
in your response, please do not include any other text than the json.

If you don't think the solution is correct, or the problem is not a good candidate for this dataset, you can set the "good_program" to False.
The tests cases should be a list of assert statements. Do not put any comments in the test cases, as they will break the json parsing.
The test cases should call the rewritten solution that you provide.
The rewritten_input should be a string similar to the original <INPUT>, but it should specify the signature of the program and the examples should use that program signature instead of the original input method.
The rewritten_input should not be a json object. It should be about the same length as the original <INPUT>, and it should mention the name of the function and the parameters it takes. If the original input includes examples, the rewritten input should include examples that use the new function signature.
{
    "rewritten_input": ...,
    "rewritten_solution": ...,
    "test_cases": [...],
    "good_program": ...,
}"""
    prompts = []
    for row in dataset:
        prompts.append((row['id'], master_prompt.replace("{input}", row['input']).replace("{solution}", row['solution'])))

    print(f"Processing {len(prompts)} prompts asynchronously...")
    results = await process_prompts(prompts)

    # Filter: make sure reference solution is correct and solves the test cases 
    url = "http://localhost:1234/test_program"
    new_results = []
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        try:
            openai_response = json.loads(result['openai_response'])
            test_cases = openai_response['test_cases']
            rewritten_solution = openai_response['rewritten_solution']
            rewritten_input = openai_response['rewritten_input']
            # Test data
            payload = {
                "program": rewritten_solution,
                "tests": test_cases,
                "max_execution_time": 2.0
            }
            # Send POST request
            response = requests.post(url, json=payload)
            response_json = response.json()
            if response.ok and sum(response_json['results']) >= 0.8 * len(test_cases):
                # throw out the test cases that are not passed
                passed_test_cases = [test_cases[j] for j in range(len(test_cases)) if response_json['results'][j] == 1]
                new_results.append({
                    "messages": [{
                        "role": "user",
                        "content": rewritten_input
                    }],
                    "ground_truth": passed_test_cases,
                    "dataset": DATASET_NAME,
                    "good_program": openai_response["good_program"],
                    "original_solution": dataset[i]['solution'],
                    "original_input": dataset[i]['input'],
                    "reasoning-output": dataset[i]['output'],
                    "source": dataset[i]['source'],
                    "difficulty": dataset[i]['difficulty'],
                    "rewritten_solution": rewritten_solution,
                    "rewritten_input": rewritten_input,
                    "id": dataset[i]['id']
                })
            else:
                print(f"Not adding. ")
                print(f"OpenAI response: {openai_response}")
                print(f"Execution results: {response_json}")
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"OpenAI response: {result['openai_response']}")
    
    print(f"After filtering, {len(new_results)} results out of {len(results)} remain")

    breakpoint()
    dataset = Dataset.from_list(new_results)
    dataset.push_to_hub(DATASET_NAME)

    print("all done")

if __name__ == "__main__":
    asyncio.run(main())