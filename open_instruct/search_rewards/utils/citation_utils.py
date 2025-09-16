import re
import os
import asyncio
from typing import Dict, List

from open_instruct.search_rewards.utils.run_utils import run_litellm, run_litellm_async


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


def extract_claims_and_corresponding_citation_ids(
    response: str, 
    split_non_cited_parts_by_newlines: bool = False,
    split_non_cited_parts_by_sentences: bool = False,
    ) -> Dict[str, List[str]]:
    """
    Example response:
    "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
    """
    claims = {}

    # Use findall to get all cite tags and their content
    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_matches = re.findall(cite_pattern, response)
    
    # Split the response by cite tags to get non-cited text
    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    non_cited_parts = re.split(cite_tag_pattern, response)
    
    # Further split non-cited parts by newlines if requested
    if split_non_cited_parts_by_newlines:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"\n", part))
        non_cited_parts = further_split_parts
    
    # Further split non-cited parts by sentences if requested
    if split_non_cited_parts_by_sentences:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"[.!?]", part))
        non_cited_parts = further_split_parts
    
    # Add non-cited text (parts between cite tags)
    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []
    
    # Add cited text with their citation IDs
    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            # When there are multiple citations (e.g., <cite id="a1b2c3d4,e5f6g7h8">), separately record the citations
            for citation_id in citation_ids.split(","):
                claims[cited_text] = [citation_id]

    return claims
    
    
def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> Dict[str, float]:
    """
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param response:
    :param citations:
    :return: final weighted score with and without the static components
    """
    if not citations:
        return 0
    
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

    claims = extract_claims_and_corresponding_citation_ids(response)

    citation_format_reward = score_citation_format(claims, citations)
    

    avg_f1 = 0
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        avg_f1 += score_citation_f1(question, claim_text, concatenated_citations, response)
    if len(claims) > 0:
        avg_f1 /= len(claims)

    return 0.6 * avg_f1 + 0.4 * citation_format_reward


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> Dict[str, float]:
    """
    Check if the model has hallucinated citations.
    """
    all_citations = []
    for claim in claims.items():
        all_citations.extend(claim[1])
    all_citations = list(set(all_citations))
    if len(all_citations) == 0:
        # If there are no citations, return 0
        return 0
    valid_citations = [citation for citation in all_citations if citation in citations]
    return len(valid_citations) / len(all_citations)


def score_citation_f1(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response)
    precision = score_citation_precision(question, claim, concatenated_citations)
    # avoid division by zero
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def score_citation_recall(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return score_no_citation_recall(question, claim, full_response)
    else:
        return score_with_citation_recall(question, claim, concatenated_citations)


def score_with_citation_recall(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = run_litellm(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return extract_recall_rating_from_response(response)


def score_no_citation_recall(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    response = run_litellm(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    # "yes" means it is a factual claim, but no citation is provided.
    return 1 - extract_yes_no_from_response(response)


def score_citation_precision(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = run_litellm(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return extract_relevant_rating_from_response(response)


def extract_recall_rating_from_response(response: str) -> float:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
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
    yes_no = re.search(r"Need Citation: \[\[(.*)\]\]", response)
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
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
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


def run_llm_judge(user_prompt: str, model_name: str = "gpt-4.1", deployment: str = "gpt-4.1-standard") -> str:
    response = run_litellm(
        model_name,
        None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


# Async versions of citation scoring functions

async def score_with_citation_recall_async(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = await run_litellm_async(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_no_citation_recall_async(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    response = await run_litellm_async(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_citation_recall_async(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return await score_no_citation_recall_async(question, claim, full_response)
    else:
        return await score_with_citation_recall_async(question, claim, concatenated_citations)


async def score_citation_precision_async(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = await run_litellm_async(
        model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_citation_f1_async(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall_task = score_citation_recall_async(question, claim, concatenated_citations, full_response)
    precision_task = score_citation_precision_async(question, claim, concatenated_citations)
    
    recall, precision = await asyncio.gather(recall_task, precision_task)
    
    # avoid division by zero
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


async def score_in_context_citations_async(question: str, response: str, citations: Dict[str, str]) -> float:
    """
    Async version of score_in_context_citations.
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param question: The original question
    :param response: The system response
    :param citations: Dictionary of citation IDs to citation text
    :return: final weighted score between 0 and 1
    """
    if not citations:
        return 0.0
    
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

    claims = extract_claims_and_corresponding_citation_ids(response)

    citation_format_reward = score_citation_format(claims, citations)
    
    # Create async tasks for all F1 score calculations
    f1_tasks = []
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        task = score_citation_f1_async(question, claim_text, concatenated_citations, response)
        f1_tasks.append(task)
    
    # Execute all F1 calculations concurrently
    if f1_tasks:
        f1_scores = await asyncio.gather(*f1_tasks)
        avg_f1 = sum(f1_scores) / len(f1_scores)
    else:
        avg_f1 = 0

    return 0.6 * avg_f1 + 0.4 * citation_format_reward
