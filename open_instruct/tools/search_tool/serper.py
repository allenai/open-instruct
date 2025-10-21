import json
import os
from typing import List

import requests
from open_instruct.tools.search_tool.s2 import create_session_with_retries


def get_snippets_for_query(query: str, api_endpoint: str = None, number_of_results: int = 10) -> List[str]:
    """
    Search using Serper.dev API for general web search and return snippets.

    Args:
        query: Search query string
        api_endpoint: Unused, kept for compatibility with SearchTool interface
        number_of_results: Number of results to return (default: 10)

    Returns:
        List containing a single string with all snippets joined by newlines
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("Missing SERPER_API_KEY environment variable.")

    session = create_session_with_retries()

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": number_of_results})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = session.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        snippets = []
        # Extract snippets from organic search results
        for result in data.get("organic", []):
            title = result.get("title", "Title not found")
            link = result.get("link", "Link not found")
            snippet = result.get("snippet", "Snippet not found")
            output_string = f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n"
            snippets.append(output_string)

        if not snippets:
            return None

        return ["\n".join(snippets)]

    except requests.exceptions.RequestException as e:
        print(f"Error performing Serper search: {e}")
        return None


if __name__ == "__main__":
    # Minimal quick test for manual running
    import sys

    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        print("SERPER_API_KEY not set; skipping live Serper test.")
        sys.exit(0)

    query = "attention is all you need"
    print(f"Running Serper quick test for query: {query!r}")
    result = get_snippets_for_query(query, number_of_results=5)
    if not result:
        print("No snippets returned or request failed.")
    else:
        print("Received snippets:")
        print(result)
