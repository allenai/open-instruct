import os

import requests

from open_instruct.search_utils.s2 import create_session_with_retries


def get_snippets_for_query(query: str, number_of_results: int = 10) -> list[str]:
    """
    Retrieves the first snippet from a web search API for the given query.
    Raises a ValueError if the API key is missing.
    """
    api_key = os.environ.get("YOUCOM_API_KEY")
    if not api_key:
        raise ValueError("Missing YOUCOM_API_KEY environment variable.")

    session = create_session_with_retries()

    headers = {"X-API-Key": api_key}
    params = {"query": query, "num_web_results": 1}
    try:
        response = session.get(
            "https://api.ydc-index.io/search",
            params=params,
            headers=headers,
            timeout=10,  # seconds
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    if "error_code" in data:
        print("Error from API:", data["error_code"])
        raise ValueError("API error occurred.")

    snippets = []
    for hit in data.get("hits", []):
        for snippet in hit.get("snippets", []):
            snippets.append(snippet)
    # Return only the first snippet if available
    return ["\n".join(["\n" + snippet for snippet in snippets[:number_of_results]])]
