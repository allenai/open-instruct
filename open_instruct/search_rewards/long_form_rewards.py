"""
Output format design.

Paper search related outputs:
- Use <S2>pes2o_id</S2> to include the search results from S2 search.

Example:


"""

import re
import logging
import hashlib
import uuid
from typing import Dict, Any, Optional

from open_instruct.search_rewards.global_rewards_utils import RubricCorpusQaGenericMetric

LOGGER = logging.getLogger(__name__)


def generate_snippet_id() -> str:
    """
    Generate a unique random ID for snippets.
    Should be used when inserting snippets into the context.
    
    Returns:
        A unique string ID in format: '<hash>'
    """
    # Generate a random UUID and create a shorter hash
    random_uuid = str(uuid.uuid4())
    hash_object = hashlib.md5(random_uuid.encode())
    short_hash = hash_object.hexdigest()[:8]  # Take first 8 characters
    return f"{short_hash}"


def extract_citations_from_context(context: str) -> Dict[str, str]:
    """
    Extract citations from the context.
    
    Citations are expected to be in the format:
    <snippets id="a1b2c3d4" metadata='{"author": "smith", "source": "arxiv", "year": 2023}'>
        Search result content here
    </snippets>
    
    The id can be any string, including hash-like IDs (e.g., a1b2c3d4)
    
    Args:
        context: The context string containing citations
        
    Returns:
        Dictionary mapping id to search results content for all found citations
    """
    citations = {}
    
    # Pattern to match <snippets id="xxx" ...> content </snippets>
    # Updated to handle quoted attributes in HTML-like format
    pattern = r'<snippets\s+id=(["\'])([^"\']+)\1[^>]*>(.*?)</snippets>'
    
    matches = re.findall(pattern, context, re.DOTALL)
    
    for quote_char, snippet_id, search_results in matches:
        # Clean up the id and search results (remove extra whitespace)
        clean_id = snippet_id.strip()
        clean_search_results = search_results.strip()
        citations[clean_id] = clean_search_results
    
    return citations


def compute_paper_reward(response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a reward score for a response based on a test case.
    
    This function:
    1. Extracts citations from the context (content before <answer> tag)
    2. Extracts the final answer from the response (content between <answer> and </answer> tags)
    3. Applies the scoring function from paper_rubrics_utils.py using the test case configuration
    
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
        'reward': 0.0,
        'citations': {},
        'answer_extracted': None,
        'extraction_success': False,
        'scoring_results': None,
        'error': None
    }
    
    try:
        # Step 1: Extract citations from the context (content before <answer> tag)
        answer_pattern = r'<answer>'
        answer_match = re.search(answer_pattern, response)
        
        if answer_match:
            context_text = response[:answer_match.start()].strip()
            extracted_citations = extract_citations_from_context(context_text)
            result['citations'] = extracted_citations
        else:
            # If no <answer> tag found, extract citations from entire response
            extracted_citations = extract_citations_from_context(response)
            result['citations'] = extracted_citations
        
        # Step 2: Extract the answer from the response
        # Pattern to match content between <answer> and </answer> tags
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        
        if not match:
            result['error'] = "Failed to extract answer from response - no <answer></answer> tags found"
            LOGGER.warning("No <answer></answer> tags found in response")
            return result
        
        extracted_answer = match.group(1).strip()
        result['answer_extracted'] = extracted_answer
        result['extraction_success'] = True
        
        # Step 3: Get the metric configuration from the test case
        if 'metric_config' not in test_case or 'config' not in test_case['metric_config']:
            result['error'] = "Invalid test case format - missing metric_config.config"
            return result
        
        metric_config = test_case['metric_config']['config']
        
        # Step 4: Initialize the scoring metric
        metric = RubricCorpusQaGenericMetric(metric_config)
        
        # Step 5: Apply the scoring function
        scoring_results = metric.score_output(extracted_answer, extracted_citations)
        
        result['scoring_results'] = scoring_results
        result['reward'] = scoring_results['score']  # Use the overall score as reward
        
        LOGGER.info(f"Successfully computed reward: {result['reward']:.4f}")
        
    except Exception as e:
        import traceback
        error_msg = f"Error computing reward: {str(e)}"
        LOGGER.error(error_msg)
        LOGGER.error(f"Full traceback: {traceback.format_exc()}")
        result['error'] = error_msg
    
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

