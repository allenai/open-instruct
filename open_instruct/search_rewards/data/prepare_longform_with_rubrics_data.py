import json
from typing import Any, Dict, List

from datasets import Dataset
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
    "dataset": "rl_rag_longform",
}
"""


system_prompt = """You are a research assistant that answers questions through iterative reasoning and research.

PROCESS:
- Use <think></think> tags to show your reasoning at any point
- Use <search>query</search> tags to write a query to search the web for information
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
<snippet id=a1b2c3d4>search results, ignored in this example</snippet>
<think>Now I need specific data on solar panel efficiency</think>
<search>latest solar panel efficiency 2024</search>
<snippet id=i9j0k1l2>search results, ignored in this example</snippet>
<answer>Based on my research... <cite id=abc123>claim from source</cite></answer>

REQUIREMENTS:
- Think and search iteratively until you have sufficient information to support your answer
- Only provide final answer when ready
- Cite all claims from search results using exact snippet IDs
- Do not cite any claims that are not from the search results"""


def upload_scholarqabench_data(use_system_prompt: bool = True) -> List[Dict[str, Any]]:
    """
    dict_keys(['initial_prompt', 'metric_config', 'case_id', 'annotator', 'agreement'])
    dict_keys(['question', 'low_length', 'high_length', 'length_weight', 'expertise_weight', 'citations_weight', 'excerpts_weight', 'other_properties'])
    """
    with open("test_configs_snippets.json", "r") as f:
        data = json.load(f)

    formatted_data = []
    for ex in data:
        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["initial_prompt"]},
            ]
        else:
            messages = [{"content": ex["initial_prompt"], "role": "user"}]
        ground_truth = json.dumps(ex)
        dataset = "rl_rag_longform"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)

    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    dataset.push_to_hub(
        "rulins/scholarqabench_rlvr_no_prompt"
        if not use_system_prompt
        else "rulins/scholarqabench_rlvr_with_system_prompt"
    )

    return formatted_data


def upload_longform_sqa_train_data(use_system_prompt: bool = True) -> List[Dict[str, Any]]:
    data = []
    with open("rubrics_1k.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    formatted_data = []
    for ex in data:
        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["Question"]},
            ]
        else:
            messages = [{"content": ex["Question"], "role": "user"}]
            
        """
        List[dict_keys(['Ingredient', 'Handle', 'Specifics'])]
        """
        ground_truth = json.dumps(ex)
        dataset = "rl_rag_longform_rubrics_only"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)
    
    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    if use_system_prompt:
        dataset.push_to_hub("rulins/rl_rag_longform_rubrics_only_with_system_prompt")
    else:
        dataset.push_to_hub("rulins/rl_rag_longform_rubrics_only_no_system_prompt")
    
    return formatted_data


def upload_longform_surveyqa_validation_data(use_system_prompt: bool = True) -> List[Dict[str, Any]]:
    dataset = load_dataset("realliyifei/ResearchQA", split="validation")

    formatted_data = []
    for ex in dataset:
        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["query"]},
            ]
        else:
            messages = [{"content": ex["query"], "role": "user"}]
        ground_truth = json.dumps(ex, default=str)
        dataset = "rl_rag_longform_rubrics_only"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)
    
    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    if use_system_prompt:
        dataset.push_to_hub("rulins/rl_rag_surveyqa_validation_longform_rubrics_only_with_system_prompt")
    else:
        dataset.push_to_hub("rulins/rl_rag_surveyqa_validation_longform_rubrics_only_no_system_prompt")
    
    return formatted_data


if __name__ == "__main__":
    # data = upload_scholarqabench_data(use_system_prompt=True)
    # data = upload_longform_sqa_train_data(use_system_prompt=True)
    data = upload_longform_surveyqa_validation_data(use_system_prompt=True)
