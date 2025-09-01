import hashlib
import uuid
from typing import Dict, Any, Optional, List
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


def extract_answer_context_citations(response: str) -> tuple[str, str, Dict[str, str]]:
    """
    Extract the answer from the context.
    
    Returns:
        tuple: (extracted_context, extracted_answer, extracted_citations)
               Returns (None, None, None) if no answer tags found
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
        return None, None, None

    extracted_answer = match.group(1).strip()
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
    pattern = r'<snippets?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets?>'

    matches = re.findall(pattern, context, re.DOTALL)

    for quote_char, snippet_id, search_results in matches:
        # Clean up the id and search results (remove extra whitespace)
        clean_id = snippet_id.strip()
        clean_search_results = search_results.strip()
        citations[clean_id] = clean_search_results

    return citations


def extract_mcp_tool_calls(context: str, mcp_parser_name: Optional[str] = None) -> List[str]:
    if not mcp_parser_name:
        return [match[1] for match in re.findall(r"<search>(.*?)</search>", context, re.DOTALL)]
    elif mcp_parser_name == "unified":
        return [match[1] for match in re.findall(r"<tool name=(.*?)>(.*?)</tool>", context, re.DOTALL)]
    elif mcp_parser_name == "v20250824":
        queries = [match[1] for match in re.findall(r"<call_tool name=(.*?)>(.*?)</call_tool>", context, re.DOTALL)]
        if not queries:
            queries = [match[1] for match in re.findall(r"<call_tool name=(.*?)>(.*?)</call>", context, re.DOTALL)]
        return queries
    else:
        raise ValueError(f"Unsupported MCP parser name: {mcp_parser_name}")
