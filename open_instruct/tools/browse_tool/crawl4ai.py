import os
from typing import Optional

from open_instruct.tools.search_tool.s2 import create_session_with_retries


def crawl_url(url: str, api_endpoint: str = None, timeout: int = 60) -> Optional[str]:
    """
    Crawl a single URL and return its markdown content.

    This is a simplified synchronous wrapper that uses the Crawl4AI Docker API.
    For more advanced features (BM25 filtering, pruning, AI2 config), use the async version.

    Args:
        url: Target URL to crawl
        api_endpoint: Base URL for the Crawl4AI Docker API (if not provided, uses CRAWL4AI_API_URL env var)
        timeout: Timeout in seconds (default: 60)

    Returns:
        String containing the markdown content of the page, or None if crawl failed
    """
    if not api_endpoint:
        api_endpoint = os.environ.get("CRAWL4AI_API_URL")
        if not api_endpoint:
            raise ValueError("CRAWL4AI_API_URL environment variable is not set or api_endpoint parameter not provided")

    api_key = os.environ.get("CRAWL4AI_API_KEY")

    session = create_session_with_retries()

    # Build the request payload
    payload = {"urls": [url], "bypass_cache": True, "ignore_links": True}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        # Call the Crawl4AI Docker API
        response = session.post(f"{api_endpoint}/crawl", json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            print(f"Error: Crawl returned no results for {url}")
            return None

        result = data[0]
        if not result.get("success", False):
            error_msg = result.get("error_message", "Unknown error")
            print(f"Error crawling {url}: {error_msg}")
            return None

        # Return the markdown content
        markdown = result.get("markdown", "")
        if not markdown:
            print(f"Warning: No markdown content returned for {url}")
            return None

        return markdown

    except Exception as e:
        print(f"Error crawling URL {url}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    result = crawl_url("https://www.example.com")
    if result:
        print(f"Successfully crawled page, got {len(result)} characters of markdown")
        print(result[:500])  # Print first 500 chars
