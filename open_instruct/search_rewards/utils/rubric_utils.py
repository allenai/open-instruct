import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from collections import defaultdict

from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_litellm, run_litellm_async
from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations

LOGGER = logging.getLogger(__name__)



def _score_property(response: str, question: str, prop: str) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion
    on a scale of 0-2.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return: score between 0 and 1 after normalizing the LLM score
    """
    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}."""
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
            return score / 2.0
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in response: {obj['score']}, error: {e}")
            return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0
    


async def _score_property_async(response: str, question: str, prop: str) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return: score between 0 and 1 after normalizing the LLM score
    """
    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score based on the given criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}."""
    user_prompt = (
        f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
    )

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = await run_litellm_async(
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
            return score / 2.0
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in response: {obj['score']}, error: {e}")
            return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0
    
    
def _score_rubric(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False) -> Dict[str, float]:
    """
    Score the response against all rubrics in the ground truth.
    
    Args:
        response: The extracted answer text to be scored
        ground_truth: Dictionary containing the question and rubrics
        
    Returns:
        Dictionary mapping rubric handles to their scores (0.0 to 1.0)
    """
    question = ground_truth["Question"] if "Question" in ground_truth else ground_truth["query"]
    
    rubric_scores = {}
    
    if use_general_rubric:
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


async def _score_weighted_rubric(response: str, ground_truth: Dict[str, Any]) -> float:
    """
    Score the response against the weighted rubric in the ground truth.
    """
    rubrics = ground_truth["rubrics"]
    question = ground_truth["query"]
    tasks = []
    for rubric in rubrics:
        task = _score_property_async(response, question, rubric["description"])
        tasks.append(task)
    scores = await asyncio.gather(*tasks)
    weighted_score = sum(score * rubric["weight"] for score, rubric in zip(scores, rubrics)) / (sum(rubric["weight"] for rubric in rubrics) + 1e-6)
    return weighted_score
    


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
You are an expert evaluator that generates adaptive rubrics for assessing model responses to a given question.

## Task
Analyze the provided question and model responses to identify key evaluation criteria that distinguish high-quality from low-quality answers.

## Output Structure
Create rubric elements with two components:
- **Ingredient**: A detailed, specific description of what makes a response excellent or problematic
- **Handle**: A concise, abstract label for the ingredient (keep general, not question-specific)

## Categories
Group elements into:
1. **Positive Rubrics**: Excellence indicators that distinguish superior responses
2. **Negative Rubrics**: Common failure patterns that degrade response quality

## Guidelines
1. **Discriminative**: Focus on criteria that clearly differentiate response quality levels
2. **Question-Relevant**: Positive rubrics should enhance helpfulness and accuracy for the specific question type
3. **Generalizable**: Negative rubrics should address common failure modes applicable across similar questions
4. **Actionable**: Each criterion should provide clear guidance for improvement
5. **Specific**: Avoid vague descriptors; be precise about what constitutes good/bad performance

## Rubric Selection Strategy
Generate 0-10 total rubrics (positive + negative combined). Decide the distribution based on what would most improve model performance:
- **More positive rubrics** when responses show potential but lack key excellence indicators
- **More negative rubrics** when responses have systematic failure patterns that need addressing
- **Balanced mix** when both strengths to amplify and weaknesses to address are present

## Response Format
Return a JSON object structured as follows containing 3 keys: "question", "positive_rubrics", "negative_rubrics":

```json
{
  "question": "<the original question>",
  "positive_rubrics": [
    {
      "ingredient": "<detailed description of excellence indicator>",
      "handle": "<concise abstract label>"
    }
  ],
  "negative_rubrics": [
    {
      "ingredient": "<detailed description of failure pattern>",
      "handle": "<concise abstract label>"
    }
  ]
}
```

## Example Rubric Elements

**Positive Example:**
```json
{
  "ingredient": "Provides step-by-step reasoning that clearly connects evidence to conclusions, making the logical flow easy to follow",
  "handle": "Clear Reasoning Chain"
}
```

**Negative Example:**
```json
{
  "ingredient": "Makes unsupported claims or presents speculation as fact without acknowledging uncertainty",
  "handle": "Unsupported Assertions"
}
```

## Input Format
You will receive the following inputs:
- **Question**: The original question that the model responses are attempting to answer
- **Responses**: Multiple model responses (Response 1, Response 2, etc.) that need evaluation
- **Existing Rubrics** (optional): Previously generated rubrics for similar questions

## Important Notes
- If existing rubrics are provided, avoid creating duplicate or highly similar rubric elements
- Focus on identifying new evaluation criteria that complement the existing rubrics
- If no new distinguishing criteria can be identified beyond existing rubrics, you may return empty arrays

Analyze the given question and responses, then generate the most impactful rubrics that capture key quality differentiators.
"""

async def generate_instance_wise_adaptive_rubrics(question, response_list, existing_rubrics=None):
    
    prompt_suffix = f"Question: {question}\n\nResponses:\n"
    for i, response in enumerate(response_list):
        prompt_suffix += f"Response {i+1}:\n{response}\n\n"
    
    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{existing_rubrics}"
    
    prompt = INSTANCE_WISE_RUBRIC_GENERATION_PROMPT + prompt_suffix
    
    resp = await run_litellm_async(
            model_name="gpt-4.1",
            user_prompt=prompt,
        )

    obj = extract_json_from_response(resp)
    
    return obj


async def _generate_instance_wise_adaptive_rubrics(responses, ground_truths, num_samples_per_prompt_rollout):
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
        
        # Get all responses for this question
        response_list = responses[start_idx:end_idx]
        answer_list = [extract_answer_context_citations(response)[1] for response in response_list]
        answer_list = [answer for answer in answer_list if answer is not None]
        # Create task for parallel execution
        task = generate_instance_wise_adaptive_rubrics(question, response_list)
        tasks.append(task)
    
    # Execute all tasks in parallel
    adaptive_rubrics = await asyncio.gather(*tasks)
    
    return adaptive_rubrics


def update_ground_truths_with_adaptive_rubrics(ground_truths, adaptive_rubrics, rubric_buffer=None):
    """
    Assume ground_truths in a format of
    {
        "query": <question>,
        "rubrics": [
            {
                "criterion": <criterion>,
                "score": <score>,
            }
        ]
    }
    Update the ground_truths with the adaptive rubrics
    """
    for ground_truth, rubrics in zip(ground_truths, adaptive_rubrics):
        if isinstance(ground_truth, list):
            # hacky fix for the data transformation that wraps the ground truth in a list
            ground_truth = ground_truth[0]
        ground_truth = json.loads(ground_truth)
        print("Ground truth: ", ground_truth)
        print("Adaptive rubrics: ", rubrics)
        positive_rubrics = rubrics["positive_rubrics"]
        negative_rubrics = rubrics["negative_rubrics"]
        ground_truth["rubrics"] = []
        for rubric in positive_rubrics:
            ground_truth["rubrics"].append({
                "criterion": rubric["ingredient"],
                "score": 1.0,
            })
        for rubric in negative_rubrics:
            ground_truth["rubrics"].append({
                "criterion": rubric["ingredient"],
                "score": -1.0,
            })
    return ground_truths