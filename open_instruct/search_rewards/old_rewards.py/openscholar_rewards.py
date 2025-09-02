import logging
import re
from typing import Any, Dict

from open_instruct.search_rewards.openscholar_rewards_utils import (
    RubricCorpusQaGenericMetric,
)
from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations

LOGGER = logging.getLogger(__name__)


def compute_paper_reward(response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a reward score for a response based on a test case.

    This function:
    1. Extracts the final answer from the response (content between <answer> and </answer> tags)
    2. Applies the scoring function from paper_rubrics_utils.py using the test case configuration

    Args:
        response: The full response text containing the answer
        test_case: A test case dictionary containing metric configuration

    Returns:
        Dictionary containing:
        - 'reward': The computed reward score (0-1)
        - 'citations': Dictionary mapping citation identifiers to text
        - 'answer_extracted': The extracted answer text
        - 'extraction_success': Boolean indicating if answer extraction was successful
        - 'scoring_results': Full scoring results from the rubric metric
        - 'error': Error message if any step failed
    """
    result = {
        "reward": 0.0,
        "citations": {},
        "answer_extracted": None,
        "extraction_success": False,
        "scoring_results": None,
        "error": None,
        "log_values": {
            "format_correct": 0.0,
        },
    }

    try:
        # Step 1: Extract answer and citations from the response
        extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
        if extracted_answer is None:
            result["error"] = "Failed to extract answer from response - no <answer></answer> tags found"
            return result
        
        result["citations"] = extracted_citations
        result["answer_extracted"] = extracted_answer
        result["context"] = extracted_context
        result["extraction_success"] = True
        result["log_values"]["format_correct"] = 1.0

        # Step 2: Get the metric configuration from the test case
        if "metric_config" not in test_case or "config" not in test_case["metric_config"]:
            result["error"] = "Invalid test case format - missing metric_config.config"
            return result

        metric_config = test_case["metric_config"]["config"]

        # Step 3: Initialize the scoring metric
        metric = RubricCorpusQaGenericMetric(metric_config)

        # Step 4: Apply the scoring function
        scoring_results = metric.score_output(extracted_answer, extracted_citations)

        result["scoring_results"] = scoring_results
        result["reward"] = scoring_results["score"]  # Use the overall score as reward
        # add the breakdowns to the log values.
        result["log_values"].update(scoring_results)

        LOGGER.info(f"Successfully computed reward: {result['reward']:.4f}")

    except Exception as e:
        import traceback

        error_msg = f"Error computing reward: {str(e)}"
        LOGGER.error(error_msg)
        LOGGER.error(f"Full traceback: {traceback.format_exc()}")
        result["error"] = error_msg

    return result


def batch_compute_paper_rewards(responses: list, test_cases: list) -> list:
    """
    Compute rewards for multiple responses and test cases.

    Args:
        responses: List of response strings
        test_cases: List of test case dictionaries

    Returns:
        List of reward result dictionaries
    """
    if len(responses) != len(test_cases):
        raise ValueError("Number of responses must match number of test cases")

    results = []
    for response, test_case in zip(responses, test_cases):
        result = compute_paper_reward(response, test_case)
        results.append(result)

    return results
