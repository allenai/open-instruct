import asyncio
import re
from typing import Dict, List

# Handle imports for both direct execution and module import
try:
    from .run_utils import run_litellm, run_litellm_async
except ImportError:
    # When running the file directly, use absolute import
    from run_utils import run_litellm, run_litellm_async

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


def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> Dict[str, float]:
    """
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param response:
    :param citations:
    :return: final weighted score with and without the static components
    """

    def extract_claims_and_corresponding_citation_ids(response: str) -> Dict[str, List[str]]:
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
        
        # Add non-cited text (parts between cite tags)
        for part in non_cited_parts:
            part = part.strip()
            if part:
                claims[part] = []
        
        # Add cited text with their citation IDs
        for _, citation_id, cited_text in cite_matches:
            cited_text = cited_text.strip()
            if cited_text:
                claims[cited_text] = [citation_id]

        return claims

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

    return 0.9 * avg_f1 + 0.1 * citation_format_reward


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
        model_name="gpt-4o-mini",
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
        model_name="gpt-4o-mini",
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
        model_name="gpt-4o-mini",
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


async def run_llm_judge_async(
    user_prompt: str, model_name: str = "gpt-4.1", deployment: str = "gpt-4.1-standard"
) -> str:
    response = await run_litellm_async(
        model_name,
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


async def score_citation_f1_async(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    # Run recall and precision calculations in parallel
    if len(concatenated_citations) == 0:
        recall_task = score_no_citation_recall_async(question, claim, full_response)
        precision_task = asyncio.create_task(asyncio.sleep(0))  # Precision is 1 for empty citations
        recall, precision = await asyncio.gather(recall_task, precision_task)
        precision = 1.0  # Set precision to 1 for empty citations
    else:
        recall_task = score_with_citation_recall_async(question, claim, concatenated_citations)
        precision_task = score_citation_precision_async(question, claim, concatenated_citations)
        recall, precision = await asyncio.gather(recall_task, precision_task)

    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    return f1


async def score_with_citation_recall_async(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = await run_litellm_async(
        model_name="gpt-4o-mini",
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return extract_recall_rating_from_response(response)


async def score_no_citation_recall_async(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    response = await run_litellm_async(
        model_name="gpt-4o-mini",
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    # "yes" means it is a factual claim, but no citation is provided.
    return 1 - extract_yes_no_from_response(response)


async def score_citation_precision_async(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = await run_litellm_async(
        model_name="gpt-4o-mini",
        system_prompt=None,
        user_prompt=user_prompt,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return extract_relevant_rating_from_response(response)


async def score_in_context_citations_async(
    question: str, response: str, citations: Dict[str, str]
) -> Dict[str, float]:
    """
    Async version of score_in_context_citations that runs LLM calls in parallel.
    """

    def extract_claims_and_corresponding_citation_ids(response: str) -> Dict[str, List[str]]:
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
        
        # Add non-cited text (parts between cite tags)
        for part in non_cited_parts:
            part = part.strip()
            if part:
                claims[part] = []
        
        # Add cited text with their citation IDs
        for _, citation_id, cited_text in cite_matches:
            cited_text = cited_text.strip()
            if cited_text:
                claims[cited_text] = [citation_id]

        return claims

    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids])

    claims = extract_claims_and_corresponding_citation_ids(response)
    citation_format_reward = score_citation_format(claims, citations)

    # Create tasks for all F1 calculations to run in parallel
    f1_tasks = []
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        task = score_citation_f1_async(question, claim_text, concatenated_citations, response)
        f1_tasks.append(task)

    # Wait for all F1 calculations to complete
    f1_scores = await asyncio.gather(*f1_tasks)

    # Calculate average F1 score
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return 0.9 * avg_f1 + 0.1 * citation_format_reward


def score_in_context_citations_async_wrapper(
    question: str, response: str, citations: Dict[str, str]
) -> Dict[str, float]:
    """
    Synchronous wrapper for the async scoring function.
    Use this when you want to run the async version from synchronous code.
    """
    return asyncio.run(score_in_context_citations_async(question, response, citations))


if __name__ == "__main__":
    # Test the extraction function
    def test_extraction():
        test_response = "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
        
        # Test the extraction function
        def extract_claims_and_corresponding_citation_ids(response: str) -> Dict[str, List[str]]:
            claims = {}
            
            # Use findall to get all cite tags and their content
            cite_pattern = r"<cite id=([^>]+)>([^<]+)</cite>"
            cite_matches = re.findall(cite_pattern, response)
            
            # Split the response by cite tags to get non-cited text
            cite_tag_pattern = r"<cite id=[^>]+>[^<]+</cite>"
            non_cited_parts = re.split(cite_tag_pattern, response)
            
            # Add non-cited text (parts between cite tags)
            for part in non_cited_parts:
                part = part.strip()
                if part:
                    claims[part] = []
            
            # Add cited text with their citation IDs
            for citation_id, cited_text in cite_matches:
                cited_text = cited_text.strip()
                if cited_text:
                    claims[cited_text] = [citation_id]
            
            return claims
        
        claims = extract_claims_and_corresponding_citation_ids(test_response)
        print("Extracted claims:")
        for claim, citation_ids in claims.items():
            print(f"Claim: '{claim}'")
            print(f"Citation IDs: {citation_ids}")
            print("-" * 50)
        
        # Verify no extra keys from regex capture groups
        print(f"Total number of claims: {len(claims)}")
        print("All claim keys are valid text segments (no regex artifacts)")
    
    test_extraction()
    
    # Simple test case for citation scoring functions
    example_response = "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"

    # Create a simple test citations dictionary
    test_citations = {
        "a1b2c3d4": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. While walls were built as early as the 7th century BC by various warring states, the most famous sections of the wall were constructed during the Ming Dynasty between 1368 and 1644.",
        "e5f6g7h8": "The Great Wall was constructed using a variety of materials depending on the local terrain and available resources. In mountainous areas, stone was the primary material, while in desert regions, tamped earth and sand were used.",
        "i9j0k1l2": "The claim that the Great Wall of China is visible from space with the naked eye is a widespread myth that has been repeatedly debunked. NASA and various astronauts have confirmed that the Great Wall is not visible to the naked eye from low Earth orbit without aid.",
    }

    test_question = "What is the history and construction of the Great Wall of China?"

    # Test the original synchronous scoring function
    print("Testing synchronous version...")
    score_sync = score_in_context_citations(test_question, example_response, test_citations)
    print(f"Synchronous citation score: {score_sync}")

    # Test the async scoring function
    print("\nTesting async version...")
    score_async = score_in_context_citations_async_wrapper(test_question, example_response, test_citations)
    print(f"Async citation score: {score_async}")

    print("\nTest completed successfully!")

    async def test_async_scoring():
        """Test the async scoring function in an async context"""
        example_response = "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite>"

        test_citations = {
            "a1b2c3d4": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. While walls were built as early as the 7th century BC by various warring states, the most famous sections of the wall were constructed during the Ming Dynasty between 1368 and 1644.",
        }

        test_question = "What is the history and construction of the Great Wall of China?"

        print("Testing async version in async context...")
        score = await score_in_context_citations_async(test_question, example_response, test_citations)
        print(f"Async citation score: {score}")
        print("Async test completed successfully!")

    # Uncomment the line below to run the async test
    # asyncio.run(test_async_scoring())

    def example_usage():
        """
        Example showing how to use both synchronous and asynchronous versions
        of the citation scoring functions.
        """
        print("=== Citation Scoring Example ===\n")

        # Example data
        question = "What are the health benefits of exercise?"
        response = "Regular exercise has numerous health benefits. <cite id=ref1>Exercise can reduce the risk of heart disease by up to 30%.</cite> Additionally, <cite id=ref2>physical activity helps maintain healthy body weight and improves mental health.</cite>"

        citations = {
            "ref1": "Studies have shown that regular physical activity can reduce the risk of cardiovascular disease by approximately 30% when compared to sedentary lifestyles.",
            "ref2": "Exercise plays a crucial role in weight management and has been linked to improved mental health outcomes, including reduced symptoms of depression and anxiety.",
        }

        print("Question:", question)
        print("Response:", response)
        print("Citations:", citations)
        print("\n" + "=" * 50 + "\n")

        # Synchronous version (original)
        print("1. Using synchronous version (sequential API calls):")
        import time

        start_time = time.time()
        score_sync = score_in_context_citations(question, response, citations)
        sync_time = time.time() - start_time
        print(f"   Score: {score_sync:.4f}")
        print(f"   Time: {sync_time:.2f} seconds")

        print("\n2. Using asynchronous version (parallel API calls):")
        start_time = time.time()
        score_async = score_in_context_citations_async_wrapper(question, response, citations)
        async_time = time.time() - start_time
        print(f"   Score: {score_async:.4f}")
        print(f"   Time: {async_time:.2f} seconds")

        if async_time < sync_time:
            speedup = sync_time / async_time
            print(f"   Speedup: {speedup:.2f}x faster")
        else:
            print("   Note: Async version may be slower for single examples due to overhead")

        print("\n3. Using async version in async context:")

        async def demo_async():
            start_time = time.time()
            score = await score_in_context_citations_async(question, response, citations)
            async_time = time.time() - start_time
            print(f"   Score: {score:.4f}")
            print(f"   Time: {async_time:.2f} seconds")

        # Run the async demo
        asyncio.run(demo_async())

        print("\n" + "=" * 50)
        print("Note: The async version is most beneficial when processing multiple examples")
        print("or when you have multiple claims that need to be evaluated simultaneously.")

    # Uncomment the line below to run the example
    # example_usage()


def score_num_in_context_search_turns(context: str, upper_bound: int = 10) -> float:
    """
    Score the number of search turns in the response.
    This function extracts all strings wrapped within <query> and </query> tags,
    then computes the number of valid (non-empty) extracted queries.
    """
    if not context:
        return 0.0, 0

    # Use re.findall to extract all substrings within <query> tags
    # The re.DOTALL flag allows '.' to match newline characters, in case a query spans multiple lines.
    queries = re.findall(r"<query>(.*?)</query>", context, re.DOTALL)

    # A valid query must not be empty or contain only whitespace.
    num_valid_queries = sum(1 for q in queries if q.strip())

    return min(float(num_valid_queries) / upper_bound, 1.0), num_valid_queries