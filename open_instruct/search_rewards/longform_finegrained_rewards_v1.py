import logging
import re
from typing import Any, Dict, Optional

from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations, compute_format_reward
from open_instruct.search_rewards.utils.rubric_utils import _score_rubric
from open_instruct.search_rewards.utils.search_utils import score_num_in_context_search_turns
from open_instruct.search_rewards.utils.citation_utils import score_in_context_citations
from open_instruct.search_rewards.utils.finegrained_utils import combine_all_reward_spans

LOGGER = logging.getLogger(__name__)


REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}


def compute_longform_finegrained_reward(response: str, ground_truth: Dict[str, Any], question: str, mcp_parser_name: Optional[str] = None, use_general_rubric: bool = False) -> Dict[str, Any]:
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
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)
    result["format_reward"] = format_reward
    
    # score num search turns
    num_search_turns_reward, num_search_turns = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    # Initialize rewards for span calculation
    rubric_reward = 0.0
    citation_reward = 0.0
    
    if extracted_answer is not None:  # only compute if answer is extracted
        # score rubric
        rubric_scores = _score_rubric(extracted_answer, ground_truth, use_general_rubric=False)
        if sum(rubric_scores.values()) == 0:
            rubric_reward = 0.0
        else:
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
        num_search_turns_reward=num_search_turns_reward,
        rubric_reward=rubric_reward,
        citation_reward=citation_reward,
        format_reward=format_reward,
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

