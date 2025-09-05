import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FinegrainedScore:
    score: float
    effective_spans: List[Tuple[int, int]]
    reward_group_id: int
    query_idx: Optional[int] = None
    response_idx: Optional[int] = None
    reward_group_name: Optional[str] = None
    advantage: Optional[float] = None


def find_format_reward_spans(response: str) -> List[Tuple[int, int]]:
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
            effective_spans=[answer_match.span()], 
            reward_group_id=0, 
            reward_group_name="answer_format",
        ))
    else:
        # Negative reward for whole response
        answer_format_reward = 0.0
        spans.append(FinegrainedScore(
            score=answer_format_reward, 
            effective_spans=[(0, len(response))], 
            reward_group_id=0, 
            reward_group_name="answer_format",
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
                effective_spans=[match.span()], 
                reward_group_id=1, 
                reward_group_name="citation_format",
            ))
    else:
        # Negative reward for whole response
        citation_format_reward = 0.0
        spans.append(FinegrainedScore(score=citation_format_reward, effective_spans=[(0, len(response))], reward_group_id=1, response_idx=None))
    
    # Check query format - reward group 2
    query_pattern = r"<query>.*?</query>"
    query_matches = list(re.finditer(query_pattern, response, re.DOTALL))
    if query_matches:
        # Positive reward for each query tag
        query_format_reward = 1.0
        for match in query_matches:
            spans.append(FinegrainedScore(
                score=query_format_reward, 
                effective_spans=[match.span()], 
                reward_group_id=2, 
                reward_group_name="query_format",
            ))
    else:
        # Negative reward for whole response
        query_format_reward = 0.0
        spans.append(FinegrainedScore(
            score=query_format_reward, 
            effective_spans=[(0, len(response))], 
            reward_group_id=2, 
            reward_group_name="query_format",
        ))
    
    return spans


def find_search_turns_reward_spans(response: str, num_search_turns_reward: float, return_span_only: bool = False) -> List["FinegrainedScore"]:
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
                    effective_spans=[(content_start, content_end)], 
                    reward_group_id=3, 
                    reward_group_name="search_turns",
                ))
        else:
            # If no query tags found but reward > 0, reward whole response
            spans.append(FinegrainedScore(
                score=num_search_turns_reward, 
                effective_spans=[(0, len(response))], 
                reward_group_id=3, 
                reward_group_name="search_turns",
            ))
    else:
        # Negative reward for whole response
        spans.append(FinegrainedScore(
            score=num_search_turns_reward, 
            effective_spans=[(0, len(response))], 
            reward_group_id=3, 
            reward_group_name="search_turns",
        ))
    
    if return_span_only:
        return spans[0].effective_spans
    else:
        return spans


def find_rubric_reward_spans(response: str, rubric_reward: float, return_span_only: bool = False) -> List["FinegrainedScore"]:
    """
    Find spans for rubric rewards.
    
    Reward/penalize the string between answer tags.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 4=rubric
    """
    spans = []
    
    # Find the answer span in the response
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    
    if answer_match:
        # Reward/penalize the content inside answer tags
        content_start = answer_match.start() + len("<answer>")
        content_end = answer_match.end() - len("</answer>")
        spans.append(FinegrainedScore(
            score=rubric_reward, 
            effective_spans=[(content_start, content_end)], 
            reward_group_id=4, 
            reward_group_name="rubric",
        ))
    else:
        # Fallback: reward/penalize whole response
        spans.append(FinegrainedScore(score=rubric_reward, effective_spans=[(0, len(response))], reward_group_id=4, reward_group_name="rubric", response_idx=None))

    if return_span_only:
        return spans[0].effective_spans
    else:
        return spans


def find_rubric_tagged_spans(response: str, tagged_sentences: List[str]) -> List[Tuple[int, int]]:
    # find the spans of the tagged sentences in the response
    spans = []
    for sentence in tagged_sentences:
        sentence_start = response.find(sentence)
        sentence_end = sentence_start + len(sentence)
        spans.append((sentence_start, sentence_end))
    return spans


def find_citation_reward_spans(response: str, citation_reward: float) -> List[Tuple[int, int]]:
    """
    Find spans for citation rewards.
    
    Reward/penalize the spans between citation tags.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_id: 5=citation
    """
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
            
            effective_spans=[(cite_start_tag_end, cite_end_tag_start)], 
    else:
        # No citations found, penalize whole response if reward is negative, or no span if positive
        if citation_reward <= 0:
            effective_spans=[(0, len(response))]
    
    return effective_spans


def combine_all_reward_spans(
    response: str,
    extracted_context: str,
    num_search_turns_reward: float,
    rubric_reward: float,
    citation_reward: float,
    format_reward: float,
) -> List["FinegrainedScore"]:
    """
    Combine all reward spans from different reward types.
    
    Returns:
        List of FinegrainedScore objects
        reward_group_ids: 0-2=format rewards, 3=search_turns, 4=rubric, 5=citation
    """
    all_rewards_with_spans = []
    
    # Format reward spans (groups 0-2)
    format_spans = find_format_reward_spans(response)  # TODO: maybe remove format from fine-grained rewards
    all_rewards_with_spans.extend(format_spans)
    
    # Search turns reward spans (group 3)
    search_spans = find_search_turns_reward_spans(response, extracted_context, num_search_turns_reward)
    all_rewards_with_spans.extend(search_spans)
    
    # Rubric reward spans (group 4)
    rubric_spans = find_rubric_reward_spans(response, rubric_reward)
    all_rewards_with_spans.extend(rubric_spans)
    
    # Citation reward spans (group 5)
    citation_spans = find_citation_reward_spans(response, citation_reward)
    citation_reward = FinegrainedScore(
        score=citation_reward,
        effective_spans=citation_spans,
        reward_group_id=5,
        reward_group_name="citation",
    )
    all_rewards_with_spans.extend(citation_spans)
    
    return all_rewards_with_spans
