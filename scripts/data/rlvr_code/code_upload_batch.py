"""
Usage:

In the root directory, run:
```
docker build -t code-api -f open_instruct/code/Dockerfile .
docker run -p 1234:1234 code-api
```

Cd into the directory of this file and run:
```
python open_code_reasoning_upload_batch.py <batch_id>
```
"""
import json
import os
import sys
from openai import AzureOpenAI, OpenAI
import requests
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from typing import List

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_HF_DATASET = "allenai/tulu-3-sft-personas-code"
OUTPUT_HF_DATASET = "saurabh5/tulu-3-personas-code-rlvr"

class OpenAIStructuredOutput(BaseModel):
    rewritten_input: str
    rewritten_solution: str
    test_cases: List[str]
    good_program: bool

def check_batch_status(batch_id: str) -> bool:
    """Check if the batch job is complete."""
    try:
        batch = client.batches.retrieve(batch_id)
        print(batch)
        print(f"Batch status: {batch.status}")
        return batch.status == "completed"
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return False

def get_batch_results(batch_id: str) -> list:
    """Retrieve and process batch results."""
    try:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(f"Batch {batch_id} is not complete. Current status: {batch.status}")
            return []

        # Download the output file
        output_file = client.files.retrieve(batch.output_file_id)
        output_content = client.files.content(output_file.id)
        
        # Get the text content from the response
        content_str = output_content.text
        
        # Process each line of the output
        results = []
        for line in content_str.split('\n'):
            if line.strip():
                try:
                    result = json.loads(line)
                    if 'response' in result and 'body' in result['response']:
                        content = result['response']['body']['choices'][0]['message']['content']
                        processed_result = json.loads(content)
                        # Extract the original ID from custom_id (format: TIMESTAMP_ID)
                        custom_id = result.get('custom_id', '')
                        if '_' in custom_id:
                            original_id = custom_id.split('_')[1]
                            processed_result['id'] = original_id
                        results.append(processed_result)
                except Exception as e:
                    print(f"Error processing result line: {e}")
        
        return results
    except Exception as e:
        print(f"Error retrieving batch results: {e}")
        return []

def process_batch_results(batch_id: str):
    """Process the batch results and upload to Hugging Face."""
    if not check_batch_status(batch_id):
        print(f"Batch {batch_id} is not complete yet")
        return

    batch_results = get_batch_results(batch_id)
    if not batch_results:
        print("No results found in batch")
        return
    
    print(f"Sample result: {batch_results[0]}")

    # Filter and validate results
    url = "http://localhost:1234/test_program"
    new_results = []
    original_dataset = load_dataset(INPUT_HF_DATASET, 'train', split="train")
    
    # Create a lookup dictionary for O(1) access
    id_to_row = {row['id']: row for row in original_dataset}
    
    for result in batch_results:
        try:
            # Look up the row directly using the ID
            if result['id'] not in id_to_row:
                print(f"No matching row found for ID: {result['id']}")
                continue
            original_dataset_row = id_to_row[result['id']]
            
            test_cases = result['test_cases']
            rewritten_solution = result['rewritten_solution']
            
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
                # Keep only passed test cases
                passed_test_cases = [test_cases[j] for j in range(len(test_cases)) if response_json['results'][j] == 1]
                
                new_results.append({
                    **original_dataset_row,
                    "messages": [{
                        "role": "user",
                        "content": result['rewritten_input']
                    }],
                    "ground_truth": passed_test_cases,
                    "dataset": OUTPUT_HF_DATASET,
                    "good_program": result["good_program"],
                    "rewritten_solution": rewritten_solution,
                    "rewritten_input": result['rewritten_input'],
                })
            else:
                print(f"Not adding. Test results: {response_json}")
        except Exception as e:
            print(f"Error processing result: {e}")

    print(f"After filtering, {len(new_results)} results out of {len(batch_results)} remain. Do you want to upload?")
    breakpoint()
    
    # Upload to Hugging Face
    if new_results:
        dataset = Dataset.from_list(new_results)
        dataset.push_to_hub(OUTPUT_HF_DATASET)
        print(f"Uploaded {len(new_results)} examples to {OUTPUT_HF_DATASET}")
    else:
        print("No valid results to upload")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python open_code_reasoning_upload_batch.py <batch_id>")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    process_batch_results(batch_id)