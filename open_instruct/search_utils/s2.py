import os
import requests

def get_snippets_for_query(query):
    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        raise ValueError("Missing S2_API_KEY environment variable.")

    res = requests.get(
        "https://api.semanticscholar.org/graph/v1/snippet/search",
        params={
            "limit": 20,
            "query": query,
        },
        headers={"x-api-key": api_key},
        timeout=60,  # long query needs more time
    )
    try:
        if res.status_code == 200:
            res = res.json()['data']
            snippets = [item['snippet']['text'] for item in res if item['snippet']]
            snippets = snippets[:2]
            return ["\n".join(snippets)]
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error: {e}")
    
    print("Failed to retrieve S2 snippets for one query.")
    return None

if __name__ == "__main__":
    print(get_snippets_for_query("colbert model retrieval"))