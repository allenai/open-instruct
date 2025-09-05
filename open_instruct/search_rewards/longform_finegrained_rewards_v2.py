"""
Longform finegrained rewards v2.
- Compared to v1, this version tags the spans for rubrics using LLM judgement.
"""
import logging
import re
from typing import Any, Dict, Optional

from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations, compute_format_reward
from open_instruct.search_rewards.utils.rubric_utils import _score_property_with_spans
from open_instruct.search_rewards.utils.search_utils import score_num_in_context_search_turns
from open_instruct.search_rewards.utils.citation_utils import score_in_context_citations
from open_instruct.search_rewards.utils.finegrained_utils import (
    FinegrainedScore,
)

LOGGER = logging.getLogger(__name__)


REWARD_WEIGHTS = {
    "rubric_reward": 0.7,
    "citation_reward": 0.1,
    "format_reward": 0.1,
    "num_search_turns_reward": 0.1,
}


def compute_longform_finegrained_reward_v2(response: str, ground_truth: Dict[str, Any], question: str, mcp_parser_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute longform finegrained reward with spans.
    
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - log_values: Dict of metrics for logging
    """
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    
    # Initialize result structure
    finegrained_scores = []
    reward_group_id_counter = 0
    reward = 0.0
    
    # score format
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)
    finegrained_scores.append(FinegrainedScore(
        score=format_reward,
        effective_spans=[(0, len(response))],  # assign format reward to the whole response
        reward_group_id=reward_group_id_counter,
        reward_group_name="format",
    ))
    reward_group_id_counter += 1
    reward += format_reward * REWARD_WEIGHTS["format_reward"]
    
    
    # score num search turns
    num_search_turns_reward, num_search_turns = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)
    finegrained_scores.append(FinegrainedScore(
        score=num_search_turns_reward,
        effective_spans=[(0, len(response))],  # assign num search turns reward to the whole response
        reward_group_id=reward_group_id_counter,
        reward_group_name="num_search_turns",
    ))
    reward_group_id_counter += 1
    reward += num_search_turns_reward * REWARD_WEIGHTS["num_search_turns_reward"]
    
    
    # score rubric
    rubric_rewards = []
    for rubric in ground_truth["Answer Critical"]:
        rubric_reward, rubric_spans_text = _score_property_with_spans(extracted_answer, question, rubric["Ingredient"])
        # get the start and end indices of the span
        rubric_spans = []
        for span_text in rubric_spans_text:
            if span_text in response:
                start_idx = response.find(span_text)
                end_idx = start_idx + len(span_text)
                rubric_spans.append((start_idx, end_idx))
        finegrained_scores.append(FinegrainedScore(
            score=rubric_reward,
            effective_spans=rubric_spans,
            reward_group_id=reward_group_id_counter,
            reward_group_name=rubric["Ingredient"],
        ))
        rubric_rewards.append(rubric_reward)
        reward_group_id_counter += 1
    finegrained_scores.append(FinegrainedScore(
        score=sum(rubric_rewards) / len(rubric_rewards),
        effective_spans=[(0, len(response))],  # assign averaged rubric reward to the whole response
        reward_group_id=reward_group_id_counter,
        reward_group_name="rubric",
    ))
    reward_group_id_counter += 1
    reward += sum(rubric_rewards) / len(rubric_rewards) * REWARD_WEIGHTS["rubric_reward"]
    
    
    # score citation
    citation_reward = score_in_context_citations(question, response, extracted_citations)
    finegrained_scores.append(FinegrainedScore(
        score=citation_reward,
        effective_spans=[(0, len(response))],  # assign citation reward to the whole response
        reward_group_id=reward_group_id_counter,
        reward_group_name="citation",
    ))
    reward_group_id_counter += 1
    reward += citation_reward * REWARD_WEIGHTS["citation_reward"]
    
    # Create log values for tracking
    log_values = {
        "format_reward": format_reward,
        "num_search_turns_reward": num_search_turns_reward,
        "rubric_reward": sum(rubric_rewards) / len(rubric_rewards),
        "citation_reward": citation_reward,
        "overall_reward": reward,  # only for logging purpose
        "response_length": len(response),
        "answer_extracted": extracted_answer is not None,
        "num_citations": len(extracted_citations) if extracted_citations else 0,
        "num_search_turns": num_search_turns,
    }
    
    return {
        "finegrained_scores": finegrained_scores,
        "log_values": log_values,
    }

