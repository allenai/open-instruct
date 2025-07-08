import asyncio
import logging
from typing import Any, Dict

# from .judge_prompts import HLE_JUDGE_PROMPT, extract_hle_judge_response_from_response
from .judge_prompts import HLE_JUDGE_SCORE_PROMPT as HLE_JUDGE_PROMPT, extract_hle_score_judge_response_from_response as extract_hle_judge_response_from_response
from .judge_prompts import HLE_JUDGE_SCORE_NO_REASONING_PROMPT as HLE_JUDGE_PROMPT_NO_REASONING
from .run_utils import run_litellm_async
from .citation_rewards_utils import score_in_context_citations
from .format_utils import extract_answer_context_citations


LOGGER = logging.getLogger(__name__)


async def hle_judge_reward_async(question: str, response: str, correct_answer: str, no_reasoning: bool = False) -> Dict[str, Any]:
    if no_reasoning:
        judge_prompt = HLE_JUDGE_PROMPT_NO_REASONING.format(question=question, response=response, correct_answer=correct_answer)
    else:
        judge_prompt = HLE_JUDGE_PROMPT.format(question=question, response=response, correct_answer=correct_answer)
    judge_response = await run_litellm_async(
        model_name="gpt-4o-mini", 
        system_prompt=None, 
        user_prompt=judge_prompt,
    )
    return judge_response


def compute_hle_reward(response: str, correct_answer: str, question: str, no_reasoning: bool = False) -> Dict[str, Any]:
    """
    Synchronous wrapper for compute_hle_reward_async.
    Returns:
        Dictionary containing:
        - 'reward': The computed reward score (0-1)
        - 'citations': Dictionary mapping citation identifiers to text
        - 'answer_extracted': The extracted answer text
        - 'extraction_success': Boolean indicating if answer extraction was successful
        - 'scoring_results': Full scoring results from the rubric metric
        - 'error': Error message if any step failed
    """
    return asyncio.run(compute_hle_reward_async(response, correct_answer, question, no_reasoning))


async def compute_hle_reward_async(response: str, correct_answer: str, question: str, no_reasoning: bool = False) -> Dict[str, Any]:
    """
    Returns:
        Dictionary containing:
        - 'reward': The computed reward score (0-1)
        - 'citations': Dictionary mapping citation identifiers to text
        - 'answer_extracted': The extracted answer text
        - 'extraction_success': Boolean indicating if answer extraction was successful
        - 'scoring_results': Full scoring results from the rubric metric
        - 'error': Error message if any step failed
    """
    result = {
        "reward": 0.0,
        "citations": {},
        "answer_extracted": None,
        "extraction_success": False,
        "scoring_results": None,
        "error": None,
        "log_values": {
            "format_correct": 0.0,
        },
    }
    
    # Step 1: Extract answer and citations from the responsez
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response, result)
    if extracted_context is None:
        result["error"] = "Failed to extract answer from response - no <answer></answer> tags found"
        return result
    
    # Step 2: Judge the response
    judge_response = await hle_judge_reward_async(question, extracted_answer, correct_answer, no_reasoning)
    LOGGER.info(f"Question: {question}")
    LOGGER.info(f"Extracted answer: {extracted_answer}")
    LOGGER.info(f"Correct answer: {correct_answer}")
    LOGGER.info(f"Judge response: {judge_response}")
    judge_reward = extract_hle_judge_response_from_response(judge_response)
    if judge_reward is None:
        result["error"] = "Failed to extract JSON from judge response"
        return result
    
    # Step 3: Score the citations
    citations_score = score_in_context_citations(question, extracted_answer, extracted_citations)
    
    # Step 4: Final score
    scoring_results = {
        "judge_score": judge_reward["reward"],
        "citations_score": citations_score,
    }
    
    result["reward"] = 0.7 * scoring_results["judge_score"] + 0.3 * scoring_results["citations_score"]
    result["log_values"] = scoring_results  
    result["scoring_results"] = scoring_results
    result["extraction_success"] = True
    result["log_values"]["format_correct"] = 1.0
    
    return result

