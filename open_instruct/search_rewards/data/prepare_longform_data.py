import json
from typing import List, Dict, Any
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


def load_scholarqabench_data(path: str) -> List[Dict[str, Any]]:
    """
    dict_keys(['initial_prompt', 'metric_config', 'case_id', 'annotator', 'agreement'])
    dict_keys(['question', 'low_length', 'high_length', 'length_weight', 'expertise_weight', 'citations_weight', 'excerpts_weight', 'other_properties'])
    """
    with open('test_configs_snippets.json', 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for ex in data:
        messages = [
            {
                "role": "user",
                "content": ex['initial_prompt']
            }
        ]
        ground_truth = json.dumps(ex)
        dataset = "rl_rag_longform"
        formatted_example = {
            "messages": messages,
            "ground_truth": ground_truth,
            "dataset": dataset
        }
        formatted_data.append(formatted_example)
    
    # push to huggingface
    dataset = Dataset.from_list(formatted_data, features=Features({
        "messages": Sequence(feature=Value(dtype="string")),
        "ground_truth": Value(dtype="string"),
        "dataset": Value(dtype="string")
    }))
    dataset.push_to_hub("rulins/scholarqabench_rlvr_no_prompt")
    
    return formatted_data




if __name__ == "__main__":
    data = load_scholarqabench_data('test_configs_snippets.json')
