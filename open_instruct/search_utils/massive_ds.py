'''
Search using the massive DS API.
Assumes you are hosting it yourself somewhere.
'''
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    allowed_methods=("GET", "POST")
):
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

def get_snippets_for_query(query, number_of_results=10):
    url = os.environ.get("MASSIVE_DS_URL")
    if not url:
        raise ValueError("Missing MASSIVE_DS_URL environment variable.")

    session = create_session_with_retries()
    
    try:
        res = session.post(
            url,
            json={"n_docs": 2, "query": query, "domains": "massive_ds"},  # domains is meaningless for now
            headers={"Content-Type": "application/json"},
            timeout=60,  # extended timeout for long queries
        )
        res.raise_for_status()  # Raises HTTPError for bad responses
        data = res.json()
        passages = data.get("results", []).get("passages", [])[0]  # passages is a list of lists
        passages = passages[:number_of_results]
        passages = ["Document: " + passage for passage in passages]
        return ["\n".join(passages)]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    print("Failed to retrieve massive ds passages for one query:", query)
    return None

if __name__ == "__main__":
    print(get_snippets_for_query("Where was Marie Curie born?"))
