"""
Verifiable: 50 HotpotQA + 50 2Wiki.
Rubric: 100 SQA.
"""

import os


LONG_FORM_PROMPT = """You are a research assistant that answers questions through iterative reasoning and research.

PROCESS:
- Use <think></think> tags to show your reasoning at any point
- Use <search>query</search> tags when you need information
- You can alternate between thinking and searching multiple times
- Only provide <answer></answer> tags when you have enough information for a complete response

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
<answer>Based on my research... <cite id=abc123>claim from source</cite></answer>

REQUIREMENTS:
- Think and search iteratively until you have sufficient information
- Only provide final answer when ready
- Cite all claims from search results using exact snippet IDs

Question: {question}
"""

SHORT_FORM_PROMPT = """You are a research assistant that answers questions through iterative reasoning and research.

PROCESS:
- Use <think></think> tags to show your reasoning at any point
- Use <search>query</search> tags when you need information
- You can alternate between thinking and searching multiple times
- Only provide <answer></answer> tags when you have enough information for a complete response
- The final answer between <answer></answer> tags should be a concise short-form answer. Do not provide any other information.

SEARCH RESULTS:
- Results appear as: <snippet id=UNIQUE_ID>content</snippet>
- Use exact snippet IDs for citations

WORKFLOW EXAMPLE:
<think>I need to understand the current market trends first</think>
<search>2024 renewable energy market trends</search>
[results provided]
<think>Now I need specific data on solar panel efficiency</think>
<search>latest solar panel efficiency 2024</search>
[results provided]
<answer>10 March</answer>

REQUIREMENTS:
- Think and search iteratively until you have sufficient information
- Only provide final answer when ready
- The final answer between <answer></answer> tags should be a concise short-form answer. Do not provide any other information.

Question: {question}
"""

def prepare_2wiki():
    orig_data_id = "rulins/2wiki_rlvr_no_prompt"
    new_data_id = "rulins/2wiki_rlvr_test"
    
    from datasets import load_dataset
    dataset = load_dataset(orig_data_id, split="test")
    
    def inject_prompt(example):
        question = example["messages"][0]["content"]
        prompt = SHORT_FORM_PROMPT.format(question=question)
        return {"messages": [{"role": "user", "content": prompt}]}
    
    dataset = dataset.map(inject_prompt)
    
    dataset.push_to_hub(new_data_id)


def prepare_hotpotqa():
    orig_data_id = "rulins/hotpotqa_rlvr_no_prompt"
    new_data_id = "rulins/hotpotqa_rlvr_test"
    
    from datasets import load_dataset
    dataset = load_dataset(orig_data_id, split="test")
    
    def inject_prompt(example):
        question = example["messages"][0]["content"]
        prompt = SHORT_FORM_PROMPT.format(question=question)
        return {"messages": [{"role": "user", "content": prompt}]}
    
    dataset = dataset.map(inject_prompt)
    
    dataset.push_to_hub(new_data_id)
    
def prepare_scholarqabench():
    orig_data_id = "rulins/scholarqabench_rlvr_with_system_prompt"
    new_data_id = "rulins/scholarqabench_rlvr_with_system_prompt"
    
    from datasets import load_dataset, DatasetDict
    
    dataset = load_dataset(orig_data_id, split="train")
    
    
    dataset_dict = DatasetDict({
        "train": dataset,
        "test": dataset
    })
    
    dataset_dict.push_to_hub(new_data_id)
    
if __name__ == "__main__":
    # prepare_2wiki()
    # prepare_hotpotqa()
    prepare_scholarqabench()
    