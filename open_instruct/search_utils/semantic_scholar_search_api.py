import os
import re
import requests
import logging


S2_API_KEY=os.environ["S2_API_KEY"]

def semantic_scholar_search_for_query(query, topk=1, max_doc_len=2048):
    snippets = get_snippets_for_query(query)
    formatted_string = format_s2_snippet_string(snippets, topk, max_doc_len)
    return formatted_string


def get_snippets_for_query(query):
    res = requests.get(
        f"https://api.semanticscholar.org/graph/v1/snippet/search",
        params={
            "limit": 20,
            "query": query,
        },
        headers={"x-api-key": S2_API_KEY},
        timeout=60,  # long query needs more time
    )
    try:
        if res.status_code == 200:
            return res.json()['data']
    except:
        pass
    
    print(f"Failed to retrieve S2 snippets for one query.")
    return None

    
def extract_s2_snippet_relevant_info(results, topk=10):
    def check_info_valid(info):
        if info['snippet'] and len(info['snippet'].split(' ')) > 20:
            return True
        else:
            return False
    
    useful_info = []
    for result in results:
        if len(useful_info) >= topk:
            break
        info = {
            'id': result['paper']['corpusId'],
            'title': result['paper']['title'],
            'snippet': result['snippet']['text'],
        }
        is_valid_info = check_info_valid(info)
        if is_valid_info:
            useful_info.append(info)
    if len(useful_info) < topk:
        print(f"Not enough context retrieved: expecting {topk}, retrieved {len(useful_info)}.")
    return useful_info


def format_s2_snippet_string(results, top_k, max_doc_len=None):
    if results is None:
        return ""
    
    relevant_info = extract_s2_snippet_relevant_info(results, top_k)
    
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        snippet = doc_info.get('snippet', "")
        
        # Clean snippet from HTML tags if any
        if snippet is not None:
            clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

            formatted_documents += f"**Document {i + 1}:**\n"
            formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
            formatted_documents += f"**Snippet:** {clean_snippet}\n"
    
    print(f"{len(formatted_documents.split(' '))} words in context.")
    if max_doc_len is not None:
        raw_doc_len = len(formatted_documents.split(' '))
        if raw_doc_len > max_doc_len:
            print(f"Documents exceeded max_doc_len, cutting from {raw_doc_len} to {max_doc_len} words.")
            formatted_documents = ' '.join(formatted_documents.split(' ')[:max_doc_len])
    
    return formatted_documents