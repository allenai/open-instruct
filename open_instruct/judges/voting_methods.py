from statistics import median, mode

from typing import List

def average_voting(scores: List[float]) -> float:
    vote = sum(scores) / len(scores) if scores else 0.0
    return vote

def median_voting(scores: List[float]) -> float:
    vote = median(scores) if scores else 0.0
    return vote

def majority_voting(scores: List[float]) -> float:
    try:
        vote = mode(scores)
    except:
        # If there’s no unique mode (e.g., all scores are different), return the average as a fallback
        vote = average_voting(scores)

    return vote

def weighted_average_voting(scores: List[float], weights: List[float]) -> float:
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length.")
    
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    
    vote = weighted_sum / total_weight if total_weight != 0 else 0.0
    return vote


AVAILABLE_VOTING_METHODS = {
    "average": average_voting,
    "median": median_voting,
    "majority": majority_voting,
    "weighted_average": weighted_average_voting,
}