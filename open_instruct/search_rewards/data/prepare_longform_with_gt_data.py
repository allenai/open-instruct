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


NEW_MCP_SYSTEM_PROMPT = """
You are a research assistant who answers questions through iterative reasoning and research.

## PROCESS:
- Use <think></think> tags to show your reasoning at any point.
- Use <call_tool name="...">query</call_tool> when you need information (see tools below).
- You can alternate between thinking and searching multiple times.
- Only provide <answer></answer> tags when you have enough information for a complete response. If the problem asks for a specific, short-form answer, you can also put the answer string in the \boxed{} format. 
- Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in <cite id="ID1,ID2">...</cite>, where id are snippet IDs from searched results (comma-separated if multiple). Use only returned snippets; never invent IDs. Avoid citing filler text - cite just the factual claim.


## SEARCH TOOLS (<call_tool name="...">query</call_tool>)
- You can use the following tools:

1. google_search 
- Purpose: general web search.
- Input via: <call_tool name="google_search">your query</call_tool>
- Output: web search snippets (see SEARCH RESULTS).
- Optional parameters 
    - gl: geolocation
    - hl: host language

2. browse_webpage 
- Purpose: open a specific URL (typically one returned by google_search) and extract readable page text as snippets. 
- Input via: <call_tool name="browse_webpage">https://example.com/article</call_tool>
- Output: webpage (see SEARCH RESULTS). 

3. snippet_search 
- Purpose: focused snippet retrieval from scientific papers
- Input via: <call_tool name="snippet_search">your query</call_tool>
- Output: snippets from existing papers (see SEARCH RESULTS). 
- Examples: <call_tool name="snippet_search" limit="8" year="2021-2025" fieldsOfStudy="Computer Science, Medicine">large language model retrieval evaluation</call_tool>
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
<call_tool name="google_search">2024 renewable energy market trends</call_tool>
<tool_output>[results provided as <snippet id=S_a1B9xQ2>...</snippet>, <snippet id=S_p0Zr41Q>...</snippet>]</tool_output>

<think>Now I need specific data on solar panel efficiency</think>
<call_tool name="snippet_search" limit="5" year="2023-2025" fieldsOfStudy="Engineering, Materials Science">latest solar panel efficiency 2024</call_tool>
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


def upload_scholarqabench_data(use_system_prompt: bool = True, use_new_mcp_system_prompt: bool = False) -> List[Dict[str, Any]]:
    """
    dict_keys(['initial_prompt', 'metric_config', 'case_id', 'annotator', 'agreement'])
    dict_keys(['question', 'low_length', 'high_length', 'length_weight', 'expertise_weight', 'citations_weight', 'excerpts_weight', 'other_properties'])
    """
    with open("open_instruct/search_rewards/data/test_configs_snippets.json", "r") as f:
        data = json.load(f)
    
    reference_answer_dict = {}
    with open("open_instruct/search_rewards/data/generated_reference_answers/reference_answers.jsonl", "r") as f:
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
                {"role": "system", "content": NEW_MCP_SYSTEM_PROMPT if use_new_mcp_system_prompt else system_prompt},
                {"role": "user", "content": ex["initial_prompt"]},
            ]
        else:
            messages = [{"content": ex["initial_prompt"], "role": "user"}]
        dataset = "rl_rag_longform_with_gt"
        formatted_example = {"messages": messages, "ground_truth": reference_answer, "dataset": dataset}
        formatted_data.append(formatted_example)

    # push to huggingface
    dataset = Dataset.from_list(formatted_data)
    if not use_system_prompt and not use_new_mcp_system_prompt:
        dataset.push_to_hub("rulins/scholarqabench_with_gt_rlvr_no_prompt")
    elif not use_system_prompt:
        dataset.push_to_hub("rulins/scholarqabench_with_gt_rlvr_with_system_prompt")
    elif not use_new_mcp_system_prompt:
        dataset.push_to_hub("rulins/scholarqabench_with_gt_rlvr_with_new_mcp_system_prompt")
    else:
        dataset.push_to_hub("rulins/scholarqabench_with_gt_rlvr_with_system_prompt")

    return formatted_data


if __name__ == "__main__":
    data = upload_scholarqabench_data(use_system_prompt=True, use_new_mcp_system_prompt=True)
