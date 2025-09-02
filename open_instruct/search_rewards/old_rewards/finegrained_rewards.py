import random
import json


def compute_finegrained_reward(prediction: str, label: str, query: str):
    """
    Dummy finegrained reward function that returns 2 scores for different spans.
    
    Args:
        prediction: The model's response text
        label: Ground truth or reference (can be JSON string)
        query: The original query/question
    
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - log_values: Dict of metrics for logging
    """
    
    # Parse label if it's JSON, otherwise use as string
    try:
        parsed_label = json.loads(label) if isinstance(label, str) else label
    except (json.JSONDecodeError, TypeError):
        parsed_label = label
    
    # Calculate span boundaries (first half and second half)
    prediction_length = len(prediction)
    if prediction_length == 0:
        # Handle empty prediction
        return {
            "finegrained_scores": [
                (0.0, (0, 0), 0, 0),  # First half: score=0.0, span=(0,0), group=0, response=0
                (0.0, (0, 0), 1, 0),  # Second half: score=0.0, span=(0,0), group=1, response=0
            ],
            "log_values": {
                "first_half_score": 0.0,
                "second_half_score": 0.0,
                "avg_score": 0.0,
                "prediction_length": 0
            }
        }
    
    mid_point = prediction_length // 2
    first_half_span = (0, mid_point)
    second_half_span = (mid_point, prediction_length)
    
    # Extract text for each span
    first_half_text = prediction[first_half_span[0]:first_half_span[1]]
    second_half_text = prediction[second_half_span[0]:second_half_span[1]]
    
    # Dummy scoring logic - you can replace this with actual reward computation
    # For demo purposes, we'll use simple heuristics:
    
    # First half score (reward group 0): Based on length and question words
    first_half_score = min(1.0, len(first_half_text) / 50.0)  # Longer is better, capped at 1.0
    if any(word in first_half_text.lower() for word in ['step', 'first', 'let', 'think']):
        first_half_score += 0.2  # Bonus for methodical language
    first_half_score = min(1.0, max(0.0, first_half_score))  # Clamp to [0, 1]
    
    # Second half score (reward group 1): Based on answer-like content
    second_half_score = min(1.0, len(second_half_text) / 30.0)  # Shorter answers preferred
    if any(word in second_half_text.lower() for word in ['answer', 'result', 'therefore', 'so']):
        second_half_score += 0.3  # Bonus for conclusion language
    if any(char.isdigit() for char in second_half_text):
        second_half_score += 0.2  # Bonus for numbers (assuming math problems)
    second_half_score = min(1.0, max(0.0, second_half_score))  # Clamp to [0, 1]
    
    # Add some randomness to make it more realistic
    first_half_score += random.uniform(-0.1, 0.1)
    second_half_score += random.uniform(-0.1, 0.1)
    first_half_score = min(1.0, max(0.0, first_half_score))
    second_half_score = min(1.0, max(0.0, second_half_score))
    
    # Create finegrained scores in the expected format
    finegrained_scores = [
        (first_half_score, first_half_span, 0, 0),    # First half: group 0 (methodology), response 0
        (second_half_score, second_half_span, 1, 0),  # Second half: group 1 (conclusion), response 0
    ]
    
    # Calculate aggregate metrics for logging
    avg_score = (first_half_score + second_half_score) / 2
    log_values = {
        "first_half_score": first_half_score,
        "second_half_score": second_half_score,
        "avg_score": avg_score,
        "prediction_length": prediction_length,
        "first_half_length": len(first_half_text),
        "second_half_length": len(second_half_text),
        "methodology_score": first_half_score,  # Group 0
        "conclusion_score": second_half_score,   # Group 1
    }
    
    return {
        "finegrained_scores": finegrained_scores,
        "log_values": log_values,
    }