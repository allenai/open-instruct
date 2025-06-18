import re
from typing import Dict, List

from run_utils import run_azure_openai


citation_recall_has_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


citation_recall_no_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{full_response}
</response>

<statement>
{statement}
</statement>"""


citation_precision_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almost unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""



def run_llm_judge(user_prompt: str, model_name: str="gpt-4.5-preview", deployment: str="gpt-4.5-preview-standard") -> str:
    response = run_azure_openai(
        model_name,
        None,
        user_prompt=user_prompt,
        deployment=deployment,
        max_completion_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0)
    return response



def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> Dict[str, float]:
    """
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param response:
    :param citations:
    :return: final weighted score with and without the static components
    """
    
    def extract_claims_and_corresponding_citation_ids(response: str) -> Dict[str, str]:
        """
        Example response:
        "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
        """
        claims = {}
        
        # Split the response by cite tags to separate cited and non-cited text
        cite_pattern = r'<cite id=([^>]+)>([^<]+)</cite>'
        parts = re.split(cite_pattern, response)
        
        for i in range(0, len(parts), 3):
            # Non-cited text (before cite tag)
            if i < len(parts) and parts[i].strip():
                non_cited_text = parts[i].strip()
                if non_cited_text:
                    claims[non_cited_text] = []
            
            # Cited text (inside cite tag)
            if i + 1 < len(parts) and i + 2 < len(parts):
                citation_id = parts[i + 1]
                cited_text = parts[i + 2].strip()
                if cited_text:
                    claims[cited_text] = [citation_id]
        
        return claims
    
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids])
    
    claims = extract_claims_and_corresponding_citation_ids(response)
    
    citation_format_reward = score_citation_format(claims, citations)
    
    avg_f1 = 0
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        avg_f1 += score_citation_f1(question, claim_text, concatenated_citations, response)
    avg_f1 /= len(claims)
    
    return 0.99 * avg_f1 + 0.01 * citation_format_reward


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> Dict[str, float]:
    """
    Check if the model has hallucinated citations.
    """
    all_citations = []
    for claim in claims.items():
        all_citations.extend(claim[1])
    all_citations = list(set(all_citations))
    if len(all_citations) == 0:
        return 1
    valid_citations = [citation for citation in all_citations if citation in citations]
    return len(valid_citations) / len(all_citations)


def score_citation_f1(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response)
    precision = score_citation_precision(question, claim, concatenated_citations)
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def score_citation_recall(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return score_no_citation_recall(question, claim, full_response)
    else:
        return score_with_citation_recall(question, claim, concatenated_citations)


def score_with_citation_recall(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(question=question, statement=claim, concatenated_cited_snippets=concatenated_citations)
    response = run_llm_judge(user_prompt)
    return extract_recall_rating_from_response(response)


def score_no_citation_recall(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(question=question, statement=claim, full_response=full_response)
    response = run_llm_judge(user_prompt)
    return extract_yes_no_from_response(response)


def score_citation_precision(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(question=question, statement=claim, concatenated_cited_snippets=concatenated_citations)
    response = run_llm_judge(user_prompt)
    return extract_relevant_rating_from_response(response)


def extract_recall_rating_from_response(response: str) -> float:
    rating = re.search(r'Rating: \[\[(.*)\]\]', response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "fully supported":
            return 1.0
        elif extracted_text == "partially supported":
            return 0.5
        elif extracted_text == "no support":
            return 0.0
        else:
            # If the extracted text doesn't match any expected values, return 0 as default
            return 0.0
    else:
        return 0.0


def extract_yes_no_from_response(response: str) -> int:
    yes_no = re.search(r'Need Citation: \[\[(.*)\]\]', response)
    if yes_no:
        extracted_text = yes_no.group(1).strip().lower()
        if extracted_text == "yes":
            return 1
        elif extracted_text == "no":
            return 0
        else:
            # If the extracted text is neither "yes" nor "no", return 0 as default
            return 0
    else:
        return 0


def extract_relevant_rating_from_response(response: str) -> int:
    rating = re.search(r'Rating: \[\[(.*)\]\]', response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "relevant":
            return 1
        elif extracted_text == "unrelevant":
            return 0
        else:
            # If the extracted text is neither "relevant" nor "unrelevant", return 0 as default
            return 0
    else:
        return 0


if __name__ == "__main__":
    # Simple test case for citation scoring functions
    example_response = "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
    
    # Create a simple test citations dictionary
    test_citations = {
        "a1b2c3d4": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. While walls were built as early as the 7th century BC by various warring states, the most famous sections of the wall were constructed during the Ming Dynasty between 1368 and 1644.",
        "e5f6g7h8": "The Great Wall was constructed using a variety of materials depending on the local terrain and available resources. In mountainous areas, stone was the primary material, while in desert regions, tamped earth and sand were used.",
        "i9j0k1l2": "The claim that the Great Wall of China is visible from space with the naked eye is a widespread myth that has been repeatedly debunked. NASA and various astronauts have confirmed that the Great Wall is not visible to the naked eye from low Earth orbit without aid."
    }
    
    test_question = "What is the history and construction of the Great Wall of China?"
    
    # Test the main scoring function
    score = score_in_context_citations(test_question, example_response, test_citations)
    print(f"Citation score: {score}")
    print("Test completed successfully!")
