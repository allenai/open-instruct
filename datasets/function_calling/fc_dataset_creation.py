import json
from datasets import load_dataset
from typing import List, Dict
import pandas as pd
import random
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


HF_HUB_CACHE = os.getenv('HF_HUB_CACHE')
HF_TOKEN = os.getenv('HF_TOKEN')
if HF_HUB_CACHE:
    os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
SYSTEM_PROMPT = [

"""

You are a helpful assistant with access to functions. When a user asks a question, directly identify and call the necessary functions to answer their query. Return the function calls in a JSON array format wrapped in triple backticks (```). Each function call should include the function name and its arguments based on the query context.

Follow these rules:
1. Identify all required functions from the available function set
2. Extract relevant parameters from the user's query
3. Structure the response as an array of function calls
4. Each function call should have a "name" and "arguments" field
5. Always wrap the JSON response in triple backticks (```)
6. Return only the function calls array without additional text or explanation

Example:
User: "What's the weather in London and Paris?"
Response:
```
[
    {"name": "get_weather", "arguments": {"city": "London"}},
    {"name": "get_weather", "arguments": {"city": "Paris"}}
]
```
""",
"""

You are a helpful assistant with access to functions. When a user asks a question, think step by step about which functions are needed and why, then make the function calls. Structure your response with clear reasoning followed by the function calls in JSON array format wrapped in triple backticks (```).

Follow these steps:
1. Analyze the user's query and break down their information needs
2. Identify which functions from the available set match these needs
3. Determine the required parameters for each function
4. Explain your reasoning process
5. Return the function calls array with "name" and "arguments" fields, wrapped in code blocks

Format your response as:
<thinking>
Step-by-step analysis of what functions are needed and why
</thinking>

<function_calls>
```
[Array of function calls in JSON format]
```
</function_calls>

Example:
User: "What's the weather in London and Paris?"

<thinking>
1. User wants weather information for two cities
2. Need to call weather function twice, once for each city
3. Required parameters: city names "London" and "Paris"
</thinking>

<function_calls>
```
[
    {"name": "get_weather", "arguments": {"city": "London"}},
    {"name": "get_weather", "arguments": {"city": "Paris"}}
]
```
</function_calls>

"""
]

def convert_to_openai_format(query: str, tools: List[Dict]) -> Dict:
    """Convert query and tools to OpenAI message format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": random.choice(SYSTEM_PROMPT) +'\n\ntools: ' + str(tools) + '\n\n' + query
            }
        ],
    }

# Load the dataset
print("Loading dataset...")
ds = load_dataset("Salesforce/xlam-function-calling-60k")
all_tools = ds['train']['tools']
# Convert to pandas for easier processing
df = pd.DataFrame(ds['train'])

# Process each row and create OpenAI format
print("Processing dataset...")
"""
processed_data = []
for _, row in df.iterrows():
    processed_item = convert_to_openai_format(row['query'], row['tools'] + )
    processed_item['answer'] = row['answers']  # Keep the answer for reference
    processed_item['num_tools_answers'] = len(row['answers'])
    processed_item['num_tools'] = len(row['tools'])
    processed_item['tools'] = len(row['tools'])
    processed_data.append(processed_item)

# Split into two datasets based on number of tools
single_tool_data = [item for item in processed_data if item['num_tools'] == 1]
multi_tool_data = [item for item in processed_data if item['num_tools'] > 1]
"""

processed_data_multi_tool = []
for _, row in df.iterrows():
    processed_item = convert_to_openai_format(row['query'], row['tools'] + '\n'.join(list(random.sample(all_tools, random.randint(5, 12)))))
    #tokens =len(tokenizer.encode(processed_item['messages'][0]['content']))
    #print(tokens)
    #processed_item['num_tokens'] = tokens
    processed_item['answer'] = row['answers']  # Keep the answer for reference
    processed_item['num_tools_answers'] = len(row['answers'])
    processed_item['num_tools'] = len(row['tools'])
    processed_item['tools'] = len(row['tools'])
    processed_data_multi_tool.append(processed_item)

from datasets import Dataset
ds_added_tool = Dataset.from_list(processed_data_multi_tool)
ds_added_tool = ds_added_tool.train_test_split(test_size=0.1)
ds_added_tool.push_to_hub("sarvam/RLVR_function_calling",private=True)


