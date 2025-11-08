import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_litellm, run_litellm_async
from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations

LOGGER = logging.getLogger(__name__)



def _score_property(response: str, question: str, prop: str, system_prompt: str = None, user_prompt: str = None, score_scale: float = 2.0) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion
    on a scale of 0-2.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return: score between 0 and 1 after normalizing the LLM score
    """
    if system_prompt is None:
        system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
    Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}."""
    if user_prompt is None:
        user_prompt = (
            f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
        )

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = run_litellm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
        )

        obj = extract_json_from_response(resp)
        if not obj:
            return 0.0
        
        # Validate that obj is a dictionary and has the expected structure
        if not isinstance(obj, dict) or "score" not in obj:
            LOGGER.warning(f"Invalid JSON structure in response: {obj}")
            return 0.0
            
        # Validate that score is a number
        try:
            score = float(obj["score"])
            return score / score_scale
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in response: {obj['score']}, error: {e}")
            return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0
    


async def _score_property_async(response: str, question: str, prop: str, system_prompt: str = None, user_prompt: str = None, score_scale: float = 2.0) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return: score between 0 and 1 after normalizing the LLM score
    """
    if system_prompt is None:
        system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion. Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}."""
    if user_prompt is None:
        user_prompt = (
        f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
    )
    
    # print("🚼 [Debug] Judge inputs: ", system_prompt, user_prompt)

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = await run_litellm_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
        )
        # print("🚼 [Debug] Judge response: ", resp)
        obj = extract_json_from_response(resp)
        if not obj:
            return 0.0
        
        # Validate that obj is a dictionary and has the expected structure
        if not isinstance(obj, dict) or "score" not in obj:
            LOGGER.warning(f"Invalid JSON structure in response: {obj}")
            return 0.0
            
        # Validate that score is a number
        try:
            score = float(obj["score"])
            return score / score_scale
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in response: {obj['score']}, error: {e}")
            return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0
    
    
def _score_rubric(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False, use_likert_rubric: bool = False) -> Dict[str, float]:
    """
    Score the response against all rubrics in the ground truth.
    
    Args:
        response: The extracted answer text to be scored
        ground_truth: Dictionary containing the question and rubrics
        
    Returns:
        Dictionary mapping rubric titles to their scores (0.0 to 1.0)
    """
    question = ground_truth["Question"] if "Question" in ground_truth else ground_truth["query"]
    
    rubric_scores = {}
    
    if use_likert_rubric:
        system_prompt = """You are an expert evaluator. Given a user prompt and a generated response, please rate the overall quality of the response on a scale of 1 to 10, where 1 is very poor and 10 is excellent.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain a single key "score" and the value should be an integer between 1 and 10.
Example response:
```json
{
"score": 8
}```"""
        user_prompt = f"""Given the following prompt, and response, please rate the overall quality of the response on a scale of 1 to 10.
<prompt>
{question}
</prompt>   
<response>
{response}
</response>
Your JSON Evaluation:"""
        score = _score_property(None, None, None, system_prompt=system_prompt, user_prompt=user_prompt, score_scale=10.0)
        rubric_scores["likert"] = score
        return rubric_scores
    elif use_general_rubric:
        general_rubric = """(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible
(2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially
(3) Factuality: There should be minimal factual errors
(4) Coherence: The discussion should stay focused and relevant to the topic"""
        score = _score_property(response, question, general_rubric)
        rubric_scores["general"] = score
        return rubric_scores
    elif "rubric" in ground_truth:
        for rubric in ground_truth["rubric"]:
            handle = rubric.get("type")
            rubric_text = rubric["rubric_item"]
            score = _score_property(response, question, rubric_text)
            rubric_scores[handle] = score
    elif "Answer Critical" in ground_truth:
        for rubric in ground_truth["Answer Critical"]:
            handle = rubric.get("Handle")
            rubric_text = rubric["Ingredient"]
            score = _score_property(response, question, rubric_text)
            rubric_scores[handle] = score
    else:
        raise ValueError(f"Unsupported rubric format found in ground truth: {ground_truth}")
    
    return rubric_scores


async def _score_weighted_rubric(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False, use_likert_rubric: bool = False) -> Tuple[List[float], List[float]]:
    """
    Score the response against rubrics and return individual scores and weights.
    
    Returns:
        Tuple of (scores, weights) where weighting can be done externally
    """
    rubrics = ground_truth["rubrics"]
    question = ground_truth["query"]
    tasks = []
    
    if use_likert_rubric:
        system_prompt = """You are an expert evaluator. Given a user prompt and a generated response, please rate the overall quality of the response on a scale of 1 to 10, where 1 is very poor and 10 is excellent.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain a single key "score" and the value should be an integer between 1 and 10.
Example response:
```json
{
"score": 8
}```"""
        user_prompt = f"""Given the following prompt, and response, please rate the overall quality of the response on a scale of 1 to 10.
<prompt>
{question}
</prompt>   
<response>
{response}
</response>
Your JSON Evaluation:"""
        task = _score_property_async(response, question, None, system_prompt=system_prompt, user_prompt=user_prompt, score_scale=10.0)
        tasks.append(task)
        weights = [1.0]
    elif use_general_rubric:
        general_rubric = """(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible
(2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially
(3) Factuality: There should be minimal factual errors
(4) Coherence: The discussion should stay focused and relevant to the topic"""
        task = _score_property_async(response, question, general_rubric)
        tasks.append(task)
        weights = [1.0]
    else:
        for rubric in rubrics:
            task = _score_property_async(response, question, rubric["description"])
            tasks.append(task)
        weights = [rubric["weight"] for rubric in rubrics]
    
    scores = await asyncio.gather(*tasks)
    return scores, weights
    


def _score_property_with_spans(response: str, question: str, prop: str) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion
    on a scale of 0-2. 
    In addition, output the spans of the response that are judged as satisfying the criterion.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return in a json format: 
        score (int) between 0 and 1 after normalizing the LLM score
        spans (list of strings) of the verbatim response that are judged as satisfying the criterion
    """
    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion. Judge only the specified aspect(s), not any other qualities of the answer. In addition, output the spans of the response that are judged as satisfying the criterion. You should output the spans in the format of verbatim snippets from the response that are judged as satisfying the criterion. Output JSON in the format: {{"score": x, "spans": [y]}}."""
    user_prompt = (
        f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
    )

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = run_litellm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
        )

        obj = extract_json_from_response(resp)
        if not obj:
            LOGGER.warning(f"No JSON object found in rubric span tag response: {resp}")
            return 0.0, []
        
        # Validate that obj is a dictionary and has the expected structure
        if not isinstance(obj, dict) or "score" not in obj or "spans" not in obj:
            LOGGER.warning(f"Invalid JSON structure in rubric span tag response: {obj}")
            return 0.0, []
            
        # Validate that score is a number
        try:
            score = float(obj["score"])
            spans = obj["spans"]
            for span in spans:
                print("Span included in the response: ", span in response)
            print("Tagged response ratio: ", len(" ".join(spans)) / len(response))
            return score / 2.0, spans
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in rubric span tag response: {obj['score']}, error: {e}")
            return 0.0, []
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0, []



INSTANCE_WISE_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a response excellent/problematic
- **Title**: Concise abstract label (general, not question-specific)

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior responses
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar responses
- Exclude generic criteria applying equally to all responses

### 2. Novelty & Non-Redundancy
With existing/ground truth rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing ones are broad
- Return empty lists if existing rubrics are comprehensive

### 3. Avoid Mirror Rubrics
Never create positive/negative versions of same criterion:
- ❌ "Provides clear explanations" + "Lacks clear explanations"
- ✅ Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- Response penalized if it exhibits ANY negative rubric behavior
- Focus on active mistakes vs missing features

## Selection Strategy

### Quantity: 1-5 total rubrics (fewer high-quality > many generic)

### Distribution Based on Response Patterns:
- **More positive**: Responses lack sophistication but avoid major errors
- **More negative**: Systematic failure patterns present
- **Balanced**: Both excellence gaps and failure modes exist
- **Empty lists**: Existing rubrics already comprehensive

## Analysis Process
1. Group responses by quality level
2. Find factors separating higher/lower clusters
3. Check if factors covered by existing rubrics
4. Select criteria with highest discriminative value

## Output Format
```json
{
  "question": "<original question verbatim>",
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```

## Examples

**Positive:**
```json
{"description": "Anticipates and addresses potential edge cases or exceptions to the main solution, demonstrating thorough problem understanding", "title": "Edge Case Handling"}
```

**Negative:**
```json
{"description": "Conflates correlation with causation when interpreting data or making recommendations", "title": "Causal Misattribution"}
```

## Inputs
1. **Question**: Original question being answered
2. **Responses**: Multiple model responses (Response 1, Response 2, etc.)
3. **Existing Rubrics** (optional): Previously generated/ground truth rubrics

## Critical Reminders
- Each rubric must distinguish between actual provided responses
- Exclude rubrics applying equally to all responses
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""


async def generate_instance_wise_adaptive_rubrics(question, response_list, existing_rubrics=None, model_name=os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1")):
    
    prompt_suffix = f"Question: {question}\n\nResponses:\n"
    for i, response in enumerate(response_list):
        prompt_suffix += f"Response {i+1}:\n{response}\n\n"
    
    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{existing_rubrics}"
    
    prompt = INSTANCE_WISE_RUBRIC_GENERATION_PROMPT + prompt_suffix
    
    try:
        resp = await run_litellm_async(
                model_name=model_name,
                user_prompt=prompt,
            )

        obj = extract_json_from_response(resp)
        print(f"Generated instance-wise adaptive rubrics: {obj}")
    except Exception as e:
        print(f"Error generating instance-wise adaptive rubrics: {e}")
        # None matching what happens if we cant extract the json from the response
        return None
    
    return obj


async def _generate_instance_wise_adaptive_rubrics(responses, ground_truths, num_samples_per_prompt_rollout, rubric_buffer=None):
    # Optimized: Use direct indexing instead of dictionary grouping
    # Responses are structured as [prompt1_resp1, prompt1_resp2, ..., prompt2_resp1, prompt2_resp2, ...]
    
    ground_truths = [json.loads(ground_truth[0]) for ground_truth in ground_truths]
    
    num_prompts = len(responses) // num_samples_per_prompt_rollout
    
    query_key = "query" if "query" in ground_truths[0] else "Question"
    assert query_key in ground_truths[0], f"Query key {query_key} not found in ground truth"
    
    # Prepare all tasks for parallel execution
    tasks = []
    for i in range(num_prompts):
        start_idx = i * num_samples_per_prompt_rollout
        end_idx = start_idx + num_samples_per_prompt_rollout
        
        # Get the question from the first ground truth in this group
        question = ground_truths[start_idx][query_key]
        if rubric_buffer is None:
            existing_rubrics = ground_truths[start_idx]["rubrics"]
            existing_rubrics_str = json.dumps(existing_rubrics)
        else:
            existing_rubrics = rubric_buffer[question]["active_rubrics"]
            existing_rubrics_str = json.dumps(existing_rubrics)
        
        # Get all responses for this question
        response_list = responses[start_idx:end_idx]
        answer_list = [extract_answer_context_citations(response)[1] for response in response_list]
        answer_list = [answer for answer in answer_list if answer is not None]
        # Create task for parallel execution
        task = generate_instance_wise_adaptive_rubrics(question, response_list, existing_rubrics_str, model_name=os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1"))
        tasks.append(task)
    
    # Execute all tasks in parallel
    adaptive_rubrics = await asyncio.gather(*tasks)
    
    return adaptive_rubrics


def update_ground_truths_with_adaptive_rubrics(ground_truths, all_adaptive_rubrics, num_samples_per_prompt_rollout, rubric_buffer=None):
    """
    Assume ground_truths in a format of
    {
        "query": <question>,
        "rubrics": [
            {
                "description": <description>,
                "weight": <weight>,
            }
        ]
    }
    Update the ground_truths with the adaptive rubrics and manage rubric buffer
    
    Returns:
        tuple: (ground_truths, valid_adaptive_rubric_rate, avg_num_ground_truths, avg_num_adaptive_rubrics, rubric_buffer)
    """
    valid_adaptive_rubric_rate = 0.0
    num_ground_truths = []
    num_adaptive_rubrics = []
    num_active_buffer_rubrics = []
    
    # Expand adaptive_rubrics to match ground_truths structure
    # Each adaptive rubric applies to num_samples_per_prompt_rollout ground truths
    expanded_adaptive_rubrics = []
    for rubric in all_adaptive_rubrics:
        for _ in range(num_samples_per_prompt_rollout):
            expanded_adaptive_rubrics.append(rubric)
    
    # Track processed queries to avoid duplicate buffer updates
    processed_queries = set()
    
    for i, (ground_truth, adaptive_rubrics) in enumerate(zip(ground_truths, expanded_adaptive_rubrics)):
        if adaptive_rubrics is None:
            continue
        
        # Handle the case where ground_truth is wrapped in a list
        is_wrapped_in_list = isinstance(ground_truth, list)
        if is_wrapped_in_list:
            # hacky fix for the data transformation that wraps the ground truth in a list
            ground_truth_str = ground_truth[0]
        else:
            ground_truth_str = ground_truth
            
        ground_truth_obj = json.loads(ground_truth_str)
        query = ground_truth_obj["query"]
        
        print(f"Ground truth: {ground_truth_obj}\nAdaptive rubrics: {adaptive_rubrics}")
        positive_rubrics = adaptive_rubrics["positive_rubrics"] if "positive_rubrics" in adaptive_rubrics else []
        negative_rubrics = adaptive_rubrics["negative_rubrics"] if "negative_rubrics" in adaptive_rubrics else []
        
        num_ground_truths.append(len(ground_truth_obj["rubrics"]))
        num_adaptive_rubrics.append(len(positive_rubrics) + len(negative_rubrics))
        
        # Update rubric buffer with newly generated adaptive rubrics (only once per query)
        if rubric_buffer is not None and query in rubric_buffer and query not in processed_queries:
            print(f"Updating rubric buffer for query {query}; before update, there is {len(rubric_buffer[query]['active_rubrics'])} active rubrics and {len(rubric_buffer[query]['inactive_rubrics'])} inactive rubrics")
            # Convert new adaptive rubrics to the buffer format
            new_active_rubrics = []
            for rubric in positive_rubrics:
                new_active_rubrics.append({
                    "description": rubric["description"],
                    "weight": 1.0,
                    "title": rubric["title"]
                })
            for rubric in negative_rubrics:
                new_active_rubrics.append({
                    "description": rubric["description"],
                    "weight": -1.0,
                    "title": rubric["title"]
                })
            
            # Append new rubrics to active_rubrics in buffer
            rubric_buffer[query]["active_rubrics"].extend(new_active_rubrics)
            num_active_buffer_rubrics.append(len(rubric_buffer[query]["active_rubrics"]))
            processed_queries.add(query)  # Mark this query as processed
            
        # Always use rubrics from buffer if available (for all rollouts of this query)
        if rubric_buffer is not None and query in rubric_buffer:
            # Keep original rubrics and append active rubrics from buffer
            ground_truth_obj["rubrics"] = rubric_buffer[query]["persistent_rubrics"] + rubric_buffer[query]["active_rubrics"]
            ground_truth_obj["rubrics_types"] = ["persistent"] * len(rubric_buffer[query]["persistent_rubrics"]) + ["adaptive"] * len(rubric_buffer[query]["active_rubrics"])
        else:
            print(f"No buffer found for query {query}, using newly generated rubrics")
            # Keep original rubrics and append newly generated adaptive rubrics
            original_rubrics = ground_truth_obj["rubrics"].copy()
            additional_rubrics = []
            for rubric in positive_rubrics:
                additional_rubrics.append({
                    "description": rubric["description"],
                    "weight": 1.0,
                    "title": rubric["title"]
                })
            for rubric in negative_rubrics:
                additional_rubrics.append({
                    "description": rubric["description"],
                    "weight": -1.0,
                    "title": rubric["title"]
                })
            ground_truth_obj["rubrics"] = original_rubrics + additional_rubrics
            ground_truth_obj["rubrics_types"] = ["persistent"] * len(original_rubrics) + ["adaptive"] * len(additional_rubrics)
        
        # Convert back to JSON string and update the original list
        updated_ground_truth_str = json.dumps(ground_truth_obj)
        if is_wrapped_in_list:
            ground_truths[i] = [updated_ground_truth_str]
        else:
            ground_truths[i] = updated_ground_truth_str
            
        valid_adaptive_rubric_rate += 1.0
    
    valid_adaptive_rubric_rate /= len(ground_truths)
    avg_num_ground_truths = sum(num_ground_truths) / len(num_ground_truths)
    avg_num_adaptive_rubrics = sum(num_adaptive_rubrics) / len(num_adaptive_rubrics)
    avg_num_active_buffer_rubrics = sum(num_active_buffer_rubrics) / len(num_active_buffer_rubrics) if num_active_buffer_rubrics else 0.0
    return ground_truths, valid_adaptive_rubric_rate, avg_num_ground_truths, avg_num_adaptive_rubrics, avg_num_active_buffer_rubrics, rubric_buffer
