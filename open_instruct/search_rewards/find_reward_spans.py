import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class FinegrainedScore:
    score: float
    effective_spans: List[Tuple[int, int]]
    reward_group_id: int
    reward_group_name: str
    query_idx: Optional[int] = None
    advantage: Optional[float] = None


def find_format_reward_spans(response: str) -> List[FinegrainedScore]:
    """
    Find spans for format rewards.
    
    For positive rewards: reward the special tokens (answer, cite, query tags)
    For negative rewards: penalize the whole response
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 0=answer_format, 1=citation_format, 2=query_format
    """
    spans = []
    
    # Check answer format - reward group 0
    answer_pattern = r"<answer>.*?</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        # Positive reward for answer tags
        answer_format_reward = 1.0
        spans.append(FinegrainedScore(
            score=answer_format_reward, 
            effective_span=answer_match.span(), 
            reward_group_id=0, 
            reward_group_name="answer_format",
            response_idx=None
        ))
    else:
        # Negative reward for whole response
        answer_format_reward = 0.0
        spans.append(FinegrainedScore(
            score=answer_format_reward, 
            effective_span=(0, len(response)), 
            reward_group_id=0, 
            reward_group_name="answer_format",
            response_idx=None
        ))
    
    # Check citation format - reward group 1
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    citation_matches = list(re.finditer(citation_pattern, response, re.DOTALL))
    if citation_matches:
        # Positive reward for each citation tag
        citation_format_reward = 1.0
        for match in citation_matches:
            spans.append(FinegrainedScore(
                score=citation_format_reward, 
                effective_span=match.span(), 
                reward_group_id=1, 
                reward_group_name="citation_format",
                response_idx=None
            ))
    else:
        # Negative reward for whole response
        citation_format_reward = 0.0
        spans.append(FinegrainedScore(score=citation_format_reward, effective_span=(0, len(response)), reward_group_id=1, response_idx=None))
    
    # Check query format - reward group 2
    query_pattern = r"<query>.*?</query>"
    query_matches = list(re.finditer(query_pattern, response, re.DOTALL))
    if query_matches:
        # Positive reward for each query tag
        query_format_reward = 1.0
        for match in query_matches:
            spans.append(FinegrainedScore(
                score=query_format_reward, 
                effective_span=match.span(), 
                reward_group_id=2, 
                reward_group_name="query_format",
                response_idx=None
            ))
    else:
        # Negative reward for whole response
        query_format_reward = 0.0
        spans.append(FinegrainedScore(
            score=query_format_reward, 
            effective_span=(0, len(response)), 
            reward_group_id=2, 
            reward_group_name="query_format",
            response_idx=None
        ))
    
    return spans


def find_search_turns_reward_spans(response: str, extracted_context: str, num_search_turns_reward: float) -> List[FinegrainedScore]:
    """
    Find spans for search turns rewards.
    
    For positive rewards: reward spans of query content (not just tags)
    For negative rewards: penalize the whole response
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 3=search_turns
    """
    spans = []
    
    if num_search_turns_reward > 0:
        # Find all query spans and reward the content inside
        query_pattern = r"<query>(.*?)</query>"
        query_matches = list(re.finditer(query_pattern, response, re.DOTALL))
        
        if query_matches:
            for match in query_matches:
                # Reward the content inside query tags (not the tags themselves)
                content_start = match.start() + len("<query>")
                content_end = match.end() - len("</query>")
                spans.append(FinegrainedScore(
                    score=num_search_turns_reward, 
                    effective_span=(content_start, content_end), 
                    reward_group_id=3, 
                    reward_group_name="search_turns",
                    response_idx=None))
        else:
            # If no query tags found but reward > 0, reward whole response
            spans.append(FinegrainedScore(
                score=num_search_turns_reward, 
                effective_span=(0, len(response)), 
                reward_group_id=3, 
                reward_group_name="search_turns",
                response_idx=None))
    else:
        # Negative reward for whole response
        spans.append(FinegrainedScore(
            score=num_search_turns_reward, 
            effective_span=(0, len(response)), 
            reward_group_id=3, 
            reward_group_name="search_turns",
            response_idx=None
        ))
    
    return spans


def find_rubric_reward_spans(response: str, extracted_answer: str, rubric_reward: float) -> List[FinegrainedScore]:
    """
    Find spans for rubric rewards.
    
    Reward/penalize the string between answer tags.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 4=rubric
    """
    spans = []
    
    if extracted_answer:
        # Find the answer span in the response
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        
        if answer_match:
            # Reward/penalize the content inside answer tags
            content_start = answer_match.start() + len("<answer>")
            content_end = answer_match.end() - len("</answer>")
            spans.append(FinegrainedScore(
                score=rubric_reward, 
                effective_span=(content_start, content_end), 
                reward_group_id=4, 
                reward_group_name="rubric",
                response_idx=None
            ))
        else:
            # Fallback: reward/penalize whole response
            spans.append(FinegrainedScore(score=rubric_reward, effective_span=(0, len(response)), reward_group_id=4, reward_group_name="rubric", response_idx=None))
    else:
        # No answer extracted, penalize whole response
        spans.append(FinegrainedScore(
            score=0.0, 
            effective_span=(0, len(response)), 
            reward_group_id=4, 
            reward_group_name="rubric",
            response_idx=None
        ))
    
    return spans


def find_citation_reward_spans(response: str, extracted_citations: Dict[str, str], citation_reward: float) -> List[FinegrainedScore]:
    """
    Find spans for citation rewards.
    
    Reward/penalize the spans between citation tags.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 5=citation
    """
    spans = []
    
    # Find all citation spans in the response
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>([^<]+)</cite>"
    citation_matches = list(re.finditer(citation_pattern, response, re.DOTALL))
    
    if citation_matches:
        # Reward/penalize each citation content
        for match in citation_matches:
            # Find the start and end of the content inside cite tags
            full_match = match.group(0)  # Full <cite...>content</cite>
            content = match.group(1)     # Just the content
            
            # Calculate positions
            cite_start_tag_end = match.start() + full_match.find('>') + 1
            cite_end_tag_start = match.end() - len("</cite>")
            
            spans.append(FinegrainedScore(
                score=citation_reward, 
                effective_span=(cite_start_tag_end, cite_end_tag_start), 
                reward_group_id=5, 
                reward_group_name="citation",
                response_idx=None
            ))
    else:
        # No citations found, penalize whole response if reward is negative, or no span if positive
        if citation_reward <= 0:
            spans.append(FinegrainedScore(
                score=citation_reward, 
                effective_span=(0, len(response)), 
                reward_group_id=5, 
                reward_group_name="citation",
                response_idx=None
            ))
    
    return spans


def combine_all_reward_spans(
    response: str,
    extracted_context: str,
    extracted_answer: str,
    extracted_citations: Dict[str, str],
    format_reward: float,
    num_search_turns_reward: float,
    rubric_reward: float,
    citation_reward: float
) -> List[FinegrainedScore]:
    """
    Combine all reward spans from different reward types.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_ids: 0-2=format rewards, 3=search_turns, 4=rubric, 5=citation
    """
    all_rewards_with_spans = []
    
    # Format reward spans (groups 0-2)
    format_spans = find_format_reward_spans(response)
    all_rewards_with_spans.extend(format_spans)
    
    # Search turns reward spans (group 3)
    search_spans = find_search_turns_reward_spans(response, extracted_context, num_search_turns_reward)
    all_rewards_with_spans.extend(search_spans)
    
    # Rubric reward spans (group 4)
    rubric_spans = find_rubric_reward_spans(response, extracted_answer, rubric_reward)
    all_rewards_with_spans.extend(rubric_spans)
    
    # Citation reward spans (group 5)
    citation_spans = find_citation_reward_spans(response, extracted_citations, citation_reward)
    all_rewards_with_spans.extend(citation_spans)
    
    return all_rewards_with_spans
