import json
from typing import Any, Dict, List

from datasets import Dataset, Features, Sequence, Value

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
            messages = [{"role": "user", "content": ex["initial_prompt"]}]
        ground_truth = json.dumps(ex)
        dataset = "rl_rag_longform"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)

    # push to huggingface
    dataset = Dataset.from_list(
        formatted_data,
        features=Features(
            {
                "messages": Sequence(feature=Value(dtype="string")),
                "ground_truth": Value(dtype="string"),
                "dataset": Value(dtype="string"),
            }
        ),
    )
    dataset.push_to_hub(
        "rulins/scholarqabench_rlvr_no_prompt"
        if not use_system_prompt
        else "rulins/scholarqabench_rlvr_with_system_prompt"
    )

    return formatted_data


if __name__ == "__main__":
    data = upload_scholarqabench_data(use_system_prompt=True)
