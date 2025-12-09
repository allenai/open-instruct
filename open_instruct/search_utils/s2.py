import logging
import os
from collections.abc import Collection

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_instruct import logger_utils

logger_utils.setup_logger("urllib3").setLevel(logging.DEBUG)


def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Collection[int] = (500, 502, 504),
    allowed_methods: Collection[str] = ("GET", "POST"),
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_snippets_for_query(query: str, number_of_results: int = 10) -> list[str] | None:
    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        raise ValueError("Missing S2_API_KEY environment variable.")

    session = create_session_with_retries()

    try:
        res = session.get(
            "https://api.semanticscholar.org/graph/v1/snippet/search",
            params={"limit": number_of_results, "query": query},
            headers={"x-api-key": api_key},
            timeout=60,  # extended timeout for long queries
        )
        res.raise_for_status()  # Raises HTTPError for bad responses
        data = res.json().get("data", [])
        snippets = [item["snippet"]["text"] for item in data if item.get("snippet")]
        return ["\n".join(snippets)]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    print("Failed to retrieve S2 snippets for one query:", query, res)
    return None


if __name__ == "__main__":
    print(get_snippets_for_query("colbert model retrieval"))
