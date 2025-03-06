import os
import requests
from typing import List, Dict, Tuple, Any


def get_snippets_for_query(query: str) -> List[str]:
    """
    Retrieves the first snippet from a web search API for the given query.
    Raises a ValueError if the API key is missing.
    """
    api_key = os.environ.get("YOUCOM_API_KEY")
    if not api_key:
        raise ValueError("Missing YOUCOM_API_KEY environment variable.")
    
    headers = {"X-API-Key": api_key}
    params = {"query": query, "num_web_results": 1}
    try:
        response = requests.get(
            "https://api.ydc-index.io/search",
            params=params,
            headers=headers,
            timeout=10  # seconds
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # Log the error as needed; returning empty list here
        return []
    
    snippets = []
    for hit in data.get("hits", []):
        for snippet in hit.get("snippets", []):
            snippets.append(snippet)
    # Return only the first snippet if available
    return snippets[:1]