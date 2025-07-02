import json
from typing import Any, Dict, List

from datasets import Dataset

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
- Cite all claims from search results using exact snippet IDs"""


def upload_scholarqabench_data(use_system_prompt: bool = True) -> List[Dict[str, Any]]:
    """
    dict_keys(['initial_prompt', 'metric_config', 'case_id', 'annotator', 'agreement'])
    dict_keys(['question', 'low_length', 'high_length', 'length_weight', 'expertise_weight', 'citations_weight', 'excerpts_weight', 'other_properties'])
    """
    with open("open_instruct/search_rewards/data/test_configs_snippets.json", "r") as f:
        data = json.load(f)
    
    reference_answer_dict = {}
    with open("generated_reference_answers/reference_answers.jsonl", "r") as f:
        reference_answers = [json.loads(line) for line in f]
        for answer in reference_answers:
            reference_answer_dict[answer["case_id"]] = answer["comprehensive_answer"]

    formatted_data = []
    for ex in data:
        if ex["case_id"] not in reference_answer_dict:
            print(f"Skipping {ex['case_id']} because it is not in the reference answer dictionary")
            continue
        reference_answer = reference_answer_dict[ex["case_id"]]
        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["initial_prompt"]},
            ]
        else:
            messages = [{"content": ex["initial_prompt"], "role": "user"}]
        dataset = "rl_rag_longform_with_gt"
        formatted_example = {"messages": messages, "ground_truth": reference_answer, "dataset": dataset}
        formatted_data.append(formatted_example)

    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    dataset.push_to_hub(
        "rulins/scholarqabench_with_gt_rlvr_no_prompt"
        if not use_system_prompt
        else "rulins/scholarqabench_with_gt_rlvr_with_system_prompt"
    )

    return formatted_data


if __name__ == "__main__":
    data = upload_scholarqabench_data(use_system_prompt=True)
