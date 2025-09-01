# load from the huggingface dataset inclusionAI/ASearcher-train-data
import json
from datasets import load_dataset


"""Data design:
{
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "ground_truth": {
        <JSON object>
    },
    "dataset": "re_search_f1",
}
"""


system_prompt = """You are a research assistant that answers questions through iterative reasoning and research.

PROCESS:
- Use <think></think> tags to show your reasoning at any point
- Use <search>query</search> tags when you need information
- You can alternate between thinking and searching multiple times
- Only provide <answer></answer> tags when you have enough information for a complete response
- The final answer between <answer></answer> tags should be a concise short-form answer. Do not provide any other information.

SEARCH RESULTS:
- Results appear as: <snippet id=UNIQUE_ID>content</snippet>
- Use exact snippet IDs for citations

CITATION FORMAT:
- In your final answer, wrap cited claims as: <cite id=SNIPPET_ID>your claim</cite>
- Example: <cite id=a1b2c3d4>Studies show 85% effectiveness rates</cite>

WORKFLOW EXAMPLE:
<think>I need to understand the current market trends first</think>
<search>2024 renewable energy market trends</search>
[results provided]
<think>Now I need specific data on solar panel efficiency</think>
<search>latest solar panel efficiency 2024</search>
[results provided]
<answer>Based on my research... <cite id=abc123>claim from source</cite> The final answer is \\boxed{{10 March}}.</answer>

REQUIREMENTS:
- Think and search iteratively until you have sufficient information
- Only provide final answer when ready
- Cite all claims from search results using exact snippet IDs
- Provide the final answer between <answer></answer> tags, use \\boxed{{}} format to wrap the concise short-form answer
- The final answer should be a concise short-form answer. Do not provide any other information.
- Question: {question}
"""




target_hf_username = "rl-rag"

# Load the two splits separately due to different schemas
# Use different cache directories to avoid schema conflicts
import tempfile
import os

cache_dir_base = tempfile.mkdtemp(prefix="dataset_base_")
cache_dir_lrm = tempfile.mkdtemp(prefix="dataset_lrm_")

dataset_base = load_dataset('json', 
                           data_files='hf://datasets/inclusionAI/ASearcher-train-data/ASearcher-Base-35k.jsonl',
                           cache_dir=cache_dir_base)
dataset_lrm = load_dataset('json', 
                          data_files='hf://datasets/inclusionAI/ASearcher-train-data/ASearcher-LRM-35k.jsonl',
                          cache_dir=cache_dir_lrm)

# create a new dataset with the short form prompt
def create_short_form_prompt_base(example):
    # For ASearcherBase35k: has aug_answer (list) and answer (list)
    if 'aug_answer' in example and example['aug_answer']:
        assert isinstance(example["aug_answer"], list), "aug_answer should be a list"
        ground_truth = example["aug_answer"]
    else:
        # Fallback to answer if aug_answer is not available or empty
        if isinstance(example["answer"], list):
            ground_truth = example["answer"]
        else:
            ground_truth = [example["answer"]]
    prompt = system_prompt.format(question=example["question"])
    return {
        "messages": [{"role": "user", "content": prompt}],
        "ground_truth": json.dumps(ground_truth),
        "dataset": "re_search_f1",
    }

def create_short_form_prompt_lrm(example):
    # For ASearcherLRM35k: answer is a string
    assert isinstance(example["answer"], str), "answer should be a string"
    ground_truth = [example["answer"]]
    prompt = system_prompt.format(question=example["question"])
    return {
        "messages": [{"role": "user", "content": prompt}],
        "ground_truth": json.dumps(ground_truth),
        "dataset": "re_search_f1",
    }

# Process each split separately
processed_base = dataset_base.map(create_short_form_prompt_base)
processed_lrm = dataset_lrm.map(create_short_form_prompt_lrm)

# Remove original columns to ensure consistent schema, keeping only the new columns
base_data = processed_base['train'].remove_columns([col for col in processed_base['train'].column_names if col not in ['messages', 'ground_truth', 'dataset']])
lrm_data = processed_lrm['train'].remove_columns([col for col in processed_lrm['train'].column_names if col not in ['messages', 'ground_truth', 'dataset']])

# Combine the processed datasets while preserving split information
from datasets import DatasetDict

# Create the final dataset with proper split names
final_dataset = DatasetDict({
    'ASearcherBase35k': base_data,
    'ASearcherLRM35k': lrm_data
})

print(f"Final dataset splits: {list(final_dataset.keys())}")
print(f"ASearcherBase35k size: {len(final_dataset['ASearcherBase35k'])}")
print(f"ASearcherLRM35k size: {len(final_dataset['ASearcherLRM35k'])}")

# Push the combined dataset
final_dataset.push_to_hub(f"{target_hf_username}/asearcher_short_form_rlvr_with_system_prompt")

# Clean up temporary cache directories
import shutil
shutil.rmtree(cache_dir_base, ignore_errors=True)
shutil.rmtree(cache_dir_lrm, ignore_errors=True)
print("Dataset processing completed successfully!")





########################################################
# Prepare validation data
########################################################
source_data_ids = [
    "rulins/2wiki_rlvr_no_prompt",
    "rulins/hotpotqa_rlvr_no_prompt",
    "rulins/tqa_rlvr_no_prompt",
    "rulins/nq_rlvr_no_prompt",
]

for source_data_id in source_data_ids:
    print(f"Processing {source_data_id}")
    dataset = load_dataset(source_data_id, split="test")
    print(f"Dataset size: {len(dataset)}")
    def inject_prompt(example):
        question = example["messages"][0]["content"]
        prompt = system_prompt.format(question=question)
        return {"messages": [{"role": "user", "content": prompt}], "ground_truth": example["ground_truth"], "dataset": "re_search_f1"}
    dataset = dataset.map(inject_prompt)
    dataset.push_to_hub(f"{target_hf_username}/{source_data_id.split('/')[-1]}_f1_test")