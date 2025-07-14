import hashlib
import uuid
from typing import Dict, Any
import re
import logging

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


def extract_answer_context_citations(response: str, result: Dict[str, Any]) -> str:
    """
    Extract the answer from the context.
    """
    extracted_answer = None
    extracted_citations = None
    extracted_context = None
    
    answer_pattern = r"<answer>"
    answer_match = re.search(answer_pattern, response)

    if answer_match:
        context_text = response[: answer_match.start()].strip()
        extracted_citations = extract_citations_from_context(context_text)
        extracted_context = context_text
    else:
        # If no <answer> tag found, extract citations from entire response
        extracted_citations = extract_citations_from_context(response)
        extracted_context = response

    # Step 2: Extract the answer from the response
    # Pattern to match content between <answer> and </answer> tags
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        # error_message = "Failed to extract answer from response - no <answer></answer> tags found"
        LOGGER.warning("No <answer></answer> tags found in response")
        result["extraction_success"] = False
        return None, None, None

    extracted_answer = match.group(1).strip()
    result["extraction_success"] = True
    result["answer_extracted"] = extracted_answer
    return extracted_context, extracted_answer, extracted_citations


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
    # Updated to handle both quoted and unquoted attributes in HTML-like format
    pattern = r'<snippets\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets>'

    matches = re.findall(pattern, context, re.DOTALL)

    for quote_char, snippet_id, search_results in matches:
        # Clean up the id and search results (remove extra whitespace)
        clean_id = snippet_id.strip()
        clean_search_results = search_results.strip()
        citations[clean_id] = clean_search_results

    return citations

