import requests
import re
from typing import List, Dict, Tuple, Any


def get_massiveds_snippets_for_query(query: str) -> List[str]:
    endpoint = "http://rulin@a100-st-p4de24xlarge-824:60747/search"
    json_data = {
        'query': query,
        "n_docs": 5,
        "domains": "MassiveDS",
    }
    headers = {"Content-Type": "application/json"}

    # Add 'http://' to the URL if it is not SSL/TLS secured, otherwise use 'https://'
    response = requests.post(endpoint, json=json_data, headers=headers)
    return response.json()


def format_massiveds_snippets(snippets):
    return "\n\n".join(snippets['results']['passages'][0])


def process_vllm_output_for_search(text: str) -> str:
    """
    Extracts a query from the given text and returns a snippet wrapped in a tag.
    If no query is found or no snippet is returned, an empty string is returned.
    """
    query_match = re.search(r"<query>(.*?)</query>", text)
    if not query_match:
        return ""
    
    query = query_match.group(1).strip()
    snippets = get_massiveds_snippets_for_query(query)
    if not snippets:
        return ""
    
    formatted_snippets_string = format_massiveds_snippets(snippets)
    return f"<snippet>{formatted_snippets_string}</snippet>"




text = "<query>Where was Marie Curie born?</query>"
output = process_vllm_output_for_search(text)
print(output)