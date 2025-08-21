import logging
import re
from typing import Any, Dict

from open_instruct.search_rewards.openscholar_rewards_utils import (
    RubricCorpusQaGenericMetric,
)
from open_instruct.search_rewards.format_utils import extract_answer_context_citations
from open_instruct.search_rewards.rubric_rewards import score_num_in_context_search_turns, _score_rubric
from open_instruct.search_rewards.citation_rewards_utils import score_in_context_citations
from open_instruct.search_rewards.find_reward_spans import combine_all_reward_spans

LOGGER = logging.getLogger(__name__)


REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}


def compute_format_reward(response: str) -> float:
    # check if response contains final answer between <answer></answer> tags
    answer_pattern = r"<answer>.*?</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        answer_format_reward = 1.0
    else:
        answer_format_reward = 0.0
    
    # check if response contains citations between <cite></cite> tags
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    citation_match = re.search(citation_pattern, response, re.DOTALL)
    if citation_match:
        citation_format_reward = 1.0
    else:
        citation_format_reward = 0.0
    
    # check if response contains at least one valid query between <query></query> tags
    query_pattern = r"<query>.*?</query>"
    query_match = re.search(query_pattern, response, re.DOTALL)
    if query_match:
        query_format_reward = 1.0
    else:
        query_format_reward = 0.0
    
    # compute weighted average of format rewards
    format_reward = 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward
    return format_reward

def compute_longform_finegrained_reward(response: str, ground_truth: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Compute longform finegrained reward with spans.
    
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - log_values: Dict of metrics for logging
    """
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    
    # Initialize result structure
    result = {
        "num_search_turns_reward": 0.0,
        "rubric_reward": 0.0,
        "citation_reward": 0.0,
        "format_reward": 0.0,
        "reward": 0.0,
    }
    
    # score format
    format_reward = compute_format_reward(response)
    result["format_reward"] = format_reward
    
    # score num search turns
    num_search_turns, num_search_turns_reward = score_num_in_context_search_turns(extracted_context)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    # Initialize rewards for span calculation
    rubric_reward = 0.0
    citation_reward = 0.0
    
    if extracted_answer is not None:  # only compute if answer is extracted
        # score rubric
        rubric_scores = _score_rubric(extracted_answer, ground_truth)
        rubric_reward = sum(rubric_scores.values()) / len(rubric_scores)
        result["rubric_reward"] = rubric_reward
        
        # score citation (include 0.1 weighted citation format reward)
        citation_reward = score_in_context_citations(question, response, extracted_citations)
        result["citation_reward"] = citation_reward
    
    # compute overall reward
    reward = 0.0
    for key, weight in REWARD_WEIGHTS.items():
        reward += weight * result[key]
    result["reward"] = reward
    
    # Generate finegrained spans
    finegrained_scores = combine_all_reward_spans(
        response=response,
        extracted_context=extracted_context,
        extracted_answer=extracted_answer,
        extracted_citations=extracted_citations,
        format_reward=format_reward,
        num_search_turns_reward=num_search_turns_reward,
        rubric_reward=rubric_reward,
        citation_reward=citation_reward
    )
    
    # Create log values for tracking
    log_values = {
        "format_reward": format_reward,
        "num_search_turns_reward": num_search_turns_reward,
        "rubric_reward": rubric_reward,
        "citation_reward": citation_reward,
        "overall_reward": reward,
        "response_length": len(response),
        "answer_extracted": extracted_answer is not None,
        "num_citations": len(extracted_citations) if extracted_citations else 0,
        "num_search_turns": num_search_turns,
    }
    
    return {
        "finegrained_scores": finegrained_scores,
        "log_values": log_values,
    }

