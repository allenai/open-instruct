import logging
import re
from typing import Any, Dict

from open_instruct.search_rewards.openscholar_rewards_utils import (
    RubricCorpusQaGenericMetric,
)
from open_instruct.search_rewards.format_utils import extract_answer_context_citations
from open_instruct.search_rewards.rubric_rewards import score_num_in_context_search_turns, _score_rubric
from open_instruct.search_rewards.citation_rewards_utils import score_in_context_citations

LOGGER = logging.getLogger(__name__)


REWARD_WEIGHTS = {
    "rubric": 0.5,
    "citation": 0.2,
    "format": 0.2,
    "num_search_turns": 0.1,
}


def compute_format_reward(response: str) -> Dict[str, Any]:
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

def compute_longform_averaged_outcome_reward(response: str, ground_truth: Dict[str, Any], question: str) -> Dict[str, Any]:
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    result = {
        "num_search_turns_reward": None,
        "rubric_scores": None,
        "citation_score": None,
        "format_score": None,
        "reward": None,
    }
    
    # score format
    format_score = compute_format_reward(response)
    result["format_score"] = format_score
    
    # score num search turns
    num_search_turns, num_search_turns_reward = score_num_in_context_search_turns(extracted_context)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    if extracted_answer is None:  # exit early if no answer is extracted
        result["reward"] = 0.0
        return result
    
    # score rubric
    rubric_scores = _score_rubric(extracted_answer, ground_truth)
    result["rubric_scores"] = rubric_scores
    
    # score citation (include 0.1 weighted citation format reward)
    citation_score = score_in_context_citations(question, response, extracted_citations)
    result["citation_score"] = citation_score
    
    # compute reward
    reward = 0.0
    for key, weight in REWARD_WEIGHTS.items():
        reward += weight * result[key]
    result["reward"] = reward
    
    return result
