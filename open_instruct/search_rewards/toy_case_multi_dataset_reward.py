import logging
import re
from typing import Any, Dict, List, Tuple, Optional, Union
import json

from open_instruct.search_rewards.find_reward_spans import FinegrainedScore
from open_instruct.math_utils import normalize_final_answer

LOGGER = logging.getLogger(__name__)


def split_response_and_get_spans(response: str, num_questions: int) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """
    Split a multi-question response into individual question responses and get their spans.
    Splits by </answer{i}> tags where i is the question number (0-indexed).
    
    Args:
        response: The full response containing multiple answers
        num_questions: Number of questions expected
        
    Returns:
        Tuple of (sub_responses, spans) where:
        - sub_responses: List of response text for each question
        - spans: List of span tuples [(start_char, end_char)] for each question
    """
    if num_questions <= 1:
        return [response], [[(0, len(response))]]
    
    sub_responses = []
    spans = []
    
    # Find split points using </answer{i}> tags
    split_points = [0]  # Start at beginning
    
    for i in range(num_questions - 1):  # We need num_questions - 1 split points
        end_tag = f"</answer{i}>"
        match = re.search(re.escape(end_tag), response)
        if match:
            # Split point is after the end tag
            split_point = match.end()
            split_points.append(split_point)
        else:
            # If tag not found, fall back to equal division for remaining questions
            remaining_questions = num_questions - i
            remaining_length = len(response) - split_points[-1]
            chars_per_remaining = remaining_length // remaining_questions
            
            for j in range(1, remaining_questions):
                split_point = split_points[-1] + j * chars_per_remaining
                split_points.append(split_point)
            break
    
    # Add end of response as final split point
    split_points.append(len(response))
    
    # Extract sub-responses and spans
    for i in range(num_questions):
        start_char = split_points[i]
        end_char = split_points[i + 1]
        
        sub_response = response[start_char:end_char]
        sub_responses.append(sub_response)
        spans.append([(start_char, end_char)])
    
    return sub_responses, spans


def extract_ground_truth_per_question(ground_truth: str) -> List[str]:
    """
    Extract ground truth answers for each individual question.
    Expects ground truth in format: {"answer": "answer1; answer2; answer3"}
    Returns: ["answer1", "answer2", "answer3"]
    """
    return [answer.strip() for answer in ground_truth.split(";")]


def extract_boxed_answer_from_response(response: str) -> str:
    """
    Extract the boxed answer from the response.
    Looks for content in \\boxed{...} format anywhere in the response.
    Returns the content inside the boxed expression, or empty string if not found.
    """
    # Look for \\boxed{...} pattern (flexible with whitespace)
    boxed_match = re.search(r"\\boxed\s*\{\s*(.*?)\s*\}", response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fallback: look for \boxed{...} (single backslash)
    boxed_match = re.search(r"\boxed\s*\{\s*(.*?)\s*\}", response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no boxed answer found, return empty string
    return ""

def verify_one_question(response: str, target: str, use_exact_match: bool = False) -> float:
    """
    Verify a single question response against ground truth.
    
    Args:
        response: The response text for one question
        ground_truth: Ground truth data for this question
        use_exact_match: If True, use exact match; if False, use contains match
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not response or not target:
        return 0.0
    
    # check if the response is a list of answers
    parsed_labels: Union[List, str]
    try:
        parsed = json.loads(target)
        parsed_labels = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        # Fallback: treat label as raw string or list-of-strings
        if isinstance(target, list):
            parsed_labels = target
        else:
            parsed_labels = [str(target).strip()]
    
    for label in parsed_labels:
        # Normalize both strings for comparison
        response_normalized = normalize_final_answer(extract_boxed_answer_from_response(response.strip()))
        target_normalized = normalize_final_answer(label.strip())
        
        if use_exact_match:
            # Exact match after normalization
            if response_normalized == target_normalized:
                return 1.0
        else:
            # Contains match - check if ground truth is contained in response
            if target_normalized.lower() in response_normalized.lower() or response_normalized.lower() in target_normalized.lower():
                return 1.0
    return 0.0



def compute_multi_question_reward(
    response: str, 
    ground_truth: str, 
    query: Optional[str] = None,
    use_exact_match: bool = False,
    reward_type: str = "finegrained",
    ) -> Dict[str, Any]:
    """
    Compute finegrained multi-question reward with simple verifiable scoring and spans.
    
    Args:
        response: The full response containing multiple answers
        ground_truth: Dictionary containing ground truth data
        use_exact_match: If True, use exact match; if False, use contains match
        
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - log_values: Dict of metrics for logging
    """
    # Get verifiable reward scores for each question
    ground_truth_per_question = extract_ground_truth_per_question(ground_truth)
    num_questions = len(ground_truth_per_question)
    
    # Split the response into individual question components for span generation
    sub_responses, spans = split_response_and_get_spans(response, num_questions)
    
    # Compute finegrained scores for each question
    finegrained_scores = []
    for i, (sub_response, gt, span) in enumerate(zip(sub_responses, ground_truth_per_question, spans)):
        sub_score = verify_one_question(sub_response, gt, use_exact_match)
    
        # Generate finegrained spans based on verifiable scores
        finegrained_scores.append(
            FinegrainedScore(
                score=sub_score,
                effective_spans=span,
                reward_group_id=i,
                reward_group_name=f"question_{i}",
            )
        )
    
    averaged_score = sum(item.score for item in finegrained_scores) / len(finegrained_scores)
    
    # Create log values for tracking
    log_values = {
        **{f"question_{i}_accuracy": item.score for i, item in enumerate(finegrained_scores)},
        "num_questions": num_questions,
        "averaged_accuracy": averaged_score,
    }
    
    if reward_type == "finegrained":
        print(f"🎀 finegrained_scores: {log_values}")
        return {
            "finegrained_scores": finegrained_scores,
            "log_values": log_values,
        }
    elif reward_type == "averaged":
        print(f"🎀 averaged_score: {averaged_score}")
        return {
            "score": averaged_score,
            "log_values": log_values,
        }
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")


if __name__ == "__main__":
    response = "Prefix random text. The answer is \\boxed{1}. The answer is \\boxed{2}. The answer is \\boxed{3}. this is extra"
    ground_truth = "1; 2; 3"
    print(compute_multi_question_reward(response, ground_truth, query=None, reward_type="averaged"))