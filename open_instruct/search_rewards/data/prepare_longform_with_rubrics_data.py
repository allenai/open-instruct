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


NEW_MCP_SYSTEM_PROMPT = """
You are a research assistant who answers questions through iterative reasoning and research.

## PROCESS:
- Use <think></think> tags to show your reasoning at any point.
- Use <tool name="...">query</tool> when you need information (see tools below).
- You can alternate between thinking and searching multiple times.
- Only provide <answer></answer> tags when you have enough information for a complete response. If the problem asks for a specific, short-form answer, you can also put the answer string in the \boxed{} format. 
- Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in <cite id="ID1,ID2">...</cite>, where id are snippet IDs from searched results (comma-separated if multiple). Use only returned snippets; never invent IDs. Avoid citing filler text - cite just the factual claim.


## SEARCH TOOLS (<tool name="...">query</tool>)
- You can use the following tools:

1. google_search 
- Purpose: general web search.
- Input via: <tool name="google_search">your query</tool>
- Output: web search snippets (see SEARCH RESULTS).
- Optional parameters 
    - gl: geolocation
    - hl: host language

2. browse_webpage 
- Purpose: open a specific URL (typically one returned by google_search) and extract readable page text as snippets. 
- Input via: <tool name="browse_webpage">https://example.com/article</tool>
- Output: webpage (see SEARCH RESULTS). 

3. snippet_search 
- Purpose: focused snippet retrieval from scientific papers
- Input via: <tool name="snippet_search">your query</tool>
- Output: snippets from existing papers (see SEARCH RESULTS). 
- Examples: <tool name="snippet_search" limit="8" year="2021-2025" fieldsOfStudy="Computer Science, Medicine">large language model retrieval evaluation</tool>
- Optional parameters 
    - limit: number of snippets to retrieve
    - year: publication year; you can use a single number (e.g., 2024) or a range (e.g., 2022-2025)
    - fieldsOfStudy: One or a comma-separated list from: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics.


## SEARCH RESULTS (<snippet id=UNIQUE_ID></snippet>, <webpage id=UNIQUE_ID>content</webpage>) 
- After you issue a <query>, we will execute it and return results as either <snippet> or <webpage> tags
- For web browsing, the searched results are represented as <webpage id=UNIQUE_ID>content</webpage>
- For snippet search, results appear as: <snippet id=UNIQUE_ID>content</snippet>
- The tool output will be wrapped in the <tool_output> tag. 


## ANSWER AND CITATION FORMAT (<answer></answer>,<cite></cite>)

- Once you collect all of the necessary information, generate the final answer, and mark your answer with answer tags: <answer></answer>. 
- If your answer is short (e.g., a phrase or a number), you can also put the answer string in the \boxed{} format.
- In your answer, wrap the supported text in <cite id="SNIPPET_ID"> ... </cite>. You have to use the exact ID from a returned <snippet id=...>...</snippet>.
- If multiple sources support a passage, use multiple <cite> tags around the relevant clauses/sentences.
- Examples 
<cite id="S17">LLMs often hallucinate on long-tail facts.</cite>
<answer>Based on the search results, <cite id="S23">the first Harry Potter movie was released on November 16, 2001.</cite>Therefore, the final answer is \boxed{November 16, 2001}.</answer>

## WORKFLOW EXAMPLE:

Below is a simple example that demonstrates the process and the correct use of tools and tags. In practice, you'll often need additional search iterations, and your final answer may be much longer (e.g., a multi-paragraph report).

Question: Give a concise update on 2024 renewable energy market trends and current commercial solar efficiency benchmarks. 

<think>I need to understand the current market trends first</think>
<tool name="google_search">2024 renewable energy market trends</tool>
<tool_output>[results provided as <snippet id=S_a1B9xQ2>...</snippet>, <snippet id=S_p0Zr41Q>...</snippet>]</tool_output>

<think>Now I need specific data on solar panel efficiency</think>
<tool name="snippet_search" limit="5" year="2023-2025" fieldsOfStudy="Engineering, Materials Science">latest solar panel efficiency 2024</tool>
<tool_output>[results provided as <snippet id=S_x4xU7dU>...</snippet>, <snippet id=S_GxA2ZLh>...</snippet>]</tool_output>

<think>I have enough to answer succinctly</think>
<answer>
Global renewables expanded rapidly in 2024, <cite ids="S_p0Zr41Q,S_GxA2ZLh">driven primarily by the growth of solar and wind energy.</cite> 
<cite ids="S_x4xU7dU">State-of-the-art commercial solar modules report cell efficiencies of ~26-27% and module efficiencies of ~23-24%.</cite>
\boxed{Solar leads 2024 renewables; top commercial module efficiency ~ 23-24\%}
</answer>


## REQUIREMENTS:
- Think and search iteratively until you have sufficient information
- Only provide the final answer when ready
- Cite all claims from search results using exact snippet IDs

Now, please try to think and search iteratively to find the answer to the following question: 
"""



HF_USERNAME = "rl-rag"

def upload_scholarqabench_data(use_system_prompt: bool = True, use_new_mcp_system_prompt: bool = False) -> List[Dict[str, Any]]:
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
                {"role": "system", "content": NEW_MCP_SYSTEM_PROMPT if use_new_mcp_system_prompt else system_prompt},
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
    if not use_system_prompt and not use_new_mcp_system_prompt:
        dataset.push_to_hub(f"{HF_USERNAME}/scholarqabench_rlvr_no_prompt")
    elif not use_system_prompt:
        dataset.push_to_hub(f"{HF_USERNAME}/scholarqabench_rlvr_with_system_prompt")
    elif not use_new_mcp_system_prompt:
        dataset.push_to_hub(f"{HF_USERNAME}/scholarqabench_rlvr_with_new_mcp_system_prompt")
    else:
        raise ValueError("Invalid system prompt and new MCP system prompt combination")

    return formatted_data


def upload_longform_sqa_train_data(use_system_prompt: bool = True, use_new_mcp_system_prompt: bool = False, reward_type: str = "rubrics_only", rubric_type: str = "no_retrieval_1k") -> List[Dict[str, Any]]:
    data = []
    with open(f"open_instruct/search_rewards/data/rubrics_{rubric_type}.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    formatted_data = []
    for ex in data:
        if isinstance(ex["Question"], list):
            continue
        if use_system_prompt:
            messages = [
                {"role": "system", "content": NEW_MCP_SYSTEM_PROMPT if use_new_mcp_system_prompt else system_prompt},
                {"role": "user", "content": ex["Question"]},
            ]
        else:
            messages = [{"content": ex["Question"], "role": "user"}]
            
        """
        List[dict_keys(['Ingredient', 'Handle', 'Specifics'])]
        """
        ground_truth = json.dumps(ex)
        if reward_type == "rubrics_only":
            dataset = "rl_rag_longform_rubrics_only"
        elif reward_type == "averaged_outcome":
            dataset = "rl_rag_longform_averaged_outcome"
        elif reward_type == "finegrained":
            dataset = "rl_rag_longform_finegrained"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)
    
    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    # Print one example
    if use_system_prompt and not use_new_mcp_system_prompt:
        if reward_type == "rubrics_only":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_rubrics_only_with_system_prompt")
        elif reward_type == "averaged_outcome":
            dataset.push_to_hub("rl-rag/rl_rag_sqa_searcharena_rubrics_web_augmented_longform_averaged_outcome_with_system_prompt")
            # dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_outcome_with_system_prompt")
        elif reward_type == "finegrained":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_finegrained_with_system_prompt")
    elif use_new_mcp_system_prompt:
        if reward_type == "rubrics_only":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_rubrics_only_with_new_mcp_system_prompt")
        elif reward_type == "averaged_outcome":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_outcome_with_new_mcp_system_prompt")
        elif reward_type == "finegrained":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_sqa_{rubric_type}_finegrained_with_new_mcp_system_prompt")
    else:
        raise ValueError("Invalid system prompt and new MCP system prompt combination")
    
    return formatted_data


def upload_longform_surveyqa_validation_data(use_system_prompt: bool = True, use_new_mcp_system_prompt: bool = False, reward_type: str = "rubrics_only") -> List[Dict[str, Any]]:
    dataset = load_dataset("realliyifei/ResearchQA", split="validation")

    formatted_data = []
    for ex in dataset:
        if use_system_prompt:
            messages = [
                {"role": "system", "content": NEW_MCP_SYSTEM_PROMPT if use_new_mcp_system_prompt else system_prompt},
                {"role": "user", "content": ex["query"]},
            ]
        else:
            messages = [{"content": ex["query"], "role": "user"}]
        ground_truth = json.dumps(ex, default=str)
        if reward_type == "rubrics_only":
            dataset = "rl_rag_longform_rubrics_only"
        elif reward_type == "averaged_outcome":
            dataset = "rl_rag_longform_averaged_outcome"
        elif reward_type == "finegrained":
            dataset = "rl_rag_longform_finegrained"
        formatted_example = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        formatted_data.append(formatted_example)
    
    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    if use_system_prompt and not use_new_mcp_system_prompt:
        if reward_type == "rubrics_only":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_rubrics_only_with_system_prompt")
        elif reward_type == "averaged_outcome":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_averaged_outcome_with_system_prompt")
        elif reward_type == "finegrained":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_finegrained_with_system_prompt")
    elif use_new_mcp_system_prompt:
        if reward_type == "rubrics_only":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_rubrics_only_with_new_mcp_system_prompt")
        elif reward_type == "averaged_outcome":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_averaged_outcome_with_new_mcp_system_prompt")
        elif reward_type == "finegrained":
            dataset.push_to_hub(f"{HF_USERNAME}/rl_rag_surveyqa_validation_longform_finegrained_with_new_mcp_system_prompt")
    else:
        raise ValueError("Invalid system prompt and new MCP system prompt combination")
    return formatted_data


if __name__ == "__main__":
    # data = upload_scholarqabench_data(use_system_prompt=True)
    data = upload_longform_sqa_train_data(use_system_prompt=True, use_new_mcp_system_prompt=True, reward_type="rubrics_only", rubric_type="searcharena_rubrics_web_augmented")
    # data = upload_longform_surveyqa_validation_data(use_system_prompt=True, reward_type="finegrained")
