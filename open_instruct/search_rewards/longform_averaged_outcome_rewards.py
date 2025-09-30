import logging
import re
from typing import Any, Dict, Optional

from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations, compute_format_reward
from open_instruct.search_rewards.utils.rubric_utils import _score_rubric, _score_weighted_rubric
from open_instruct.search_rewards.utils.citation_utils import score_in_context_citations, score_in_context_citations_async
from open_instruct.search_rewards.utils.search_utils import score_num_in_context_search_turns

LOGGER = logging.getLogger(__name__)


REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}

REWARD_WEIGHTS_WITHOUT_CITATION = {
    "rubric_reward": 0.6,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.2,
}

def compute_longform_averaged_outcome_reward(
        response: str, 
        ground_truth: Dict[str, Any], 
        question: str, 
        mcp_parser_name: Optional[str] = None, 
        use_general_rubric: bool = False,
        no_citation_reward: bool = False,
        use_likert_rubric: bool = False,
    ) -> Dict[str, Any]:
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
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
    num_search_turns_reward, num_search_turns  = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    if extracted_answer is None:  # exit early if no answer is extracted
        result["reward"] = 0.0
        return result
    
    # score rubric
    rubric_scores = _score_rubric(extracted_answer, ground_truth, use_general_rubric=use_general_rubric, use_likert_rubric=use_likert_rubric)
    if len(rubric_scores) == 0:
        print("🔥 No rubric scores found")
        rubric_reward = 0.0
    else:
        rubric_reward = sum(rubric_scores.values()) / len(rubric_scores)
    result["rubric_reward"] = rubric_reward
    
    # score citation (include 0.1 weighted citation format reward)
    if not no_citation_reward:
        citation_reward = score_in_context_citations(question, response, extracted_citations)
    else:
        citation_reward = 0.0
    result["citation_reward"] = citation_reward
    
    # compute reward
    reward = 0.0
    if no_citation_reward:
        weights = REWARD_WEIGHTS_WITHOUT_CITATION
    else:
        weights = REWARD_WEIGHTS
    for key, weight in weights.items():
        reward += weight * result[key]
    result["reward"] = reward
    
    return result


async def compute_longform_averaged_outcome_reward_async(
        response: str, 
        ground_truth: Dict[str, Any], 
        question: str, mcp_parser_name: Optional[str] = None, 
        use_general_rubric: bool = False,
        no_citation_reward: bool = False,
        use_likert_rubric: bool = False,
        use_full_response_as_answer: bool = False,
    ) -> Dict[str, Any]:
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response, use_full_response_as_answer=use_full_response_as_answer)
    
    result = {
        "num_search_turns_reward": 0.0,
        "rubric_reward": 0.0,
        "citation_reward": 0.0,
        "format_reward": 0.0,
        "reward": 0.0,
    }
    
    # score format
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name, use_full_response_as_answer=use_full_response_as_answer)
    result["format_reward"] = format_reward
    
    # score num search turns
    num_search_turns_reward, num_search_turns  = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    if extracted_answer is None:  # exit early if no answer is extracted
        return result
    
    # score rubric
    scores, weights = await _score_weighted_rubric(extracted_answer, ground_truth, use_general_rubric=use_general_rubric, use_likert_rubric=use_likert_rubric)
    if len(scores) == 0:
        print("🔥 No rubric scores found. This should not happen.")
        rubric_reward = 0.0
    else:
        # Compute weighted average externally
        total_weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weight for weight in weights if weight > 0)
        rubric_reward = total_weighted_score / max(total_weight, 1.0)
    result["rubric_reward"] = rubric_reward
    
    if use_full_response_as_answer:
        result["reward"] = rubric_reward
        return result
    
    # score citation (include 0.1 weighted citation format reward)
    if not no_citation_reward:
        citation_reward = score_in_context_citations(question, response, extracted_citations)
    else:
        citation_reward = 0.0
    result["citation_reward"] = citation_reward
    
    # compute reward
    reward = 0.0
    if no_citation_reward:
        weights = REWARD_WEIGHTS_WITHOUT_CITATION
    else:
        weights = REWARD_WEIGHTS
    for key, weight in weights.items():
        reward += weight * result[key]
    result["reward"] = reward
    
    return result