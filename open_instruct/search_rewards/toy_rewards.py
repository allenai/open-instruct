import json
import sys
import os
from typing import Any, Dict, Optional

# Try to import from the package, fall back to relative imports
try:
    from open_instruct.search_rewards.rubric_rewards import compute_rubric_reward
    from open_instruct.search_rewards.citation_rewards_utils import score_in_context_citations
    from open_instruct.search_rewards.format_utils import extract_answer_context_citations
    USING_REAL_FUNCTIONS = True
except ImportError:
    try:
        # Add the parent directory to path for relative imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from rubric_rewards import compute_rubric_reward
        from citation_rewards_utils import score_in_context_citations
        from format_utils import extract_answer_context_citations
        USING_REAL_FUNCTIONS = True
    except ImportError:
        # If both fail, we'll define stub functions for demonstration
        USING_REAL_FUNCTIONS = False
        
        def compute_rubric_reward(prediction: str, parsed_label: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "reward": 0.5,  # Dummy score
                "rubric_scores": {"dummy_rubric": 0.5},
                "extraction_success": True,
                "log_values": {"format_correct_has_answer": 1.0, "rubric_averaged_reward": 0.5},
                "error": None
            }
        
        def score_in_context_citations(query: str, answer: str, citations: Dict[str, str]) -> float:
            return 0.7  # Dummy citation score
        
        def extract_answer_context_citations(prediction: str, result: Dict[str, Any]):
            # Simple extraction for demo
            import re
            context_match = re.search(r'<context>(.*?)</context>', prediction, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
            
            context = context_match.group(1).strip() if context_match else ""
            answer = answer_match.group(1).strip() if answer_match else prediction
            
            # Extract citations from snippets
            citations = {}
            snippet_matches = re.findall(r'<snippets id="([^"]+)">(.*?)</snippets>', context, re.DOTALL)
            for snippet_id, snippet_content in snippet_matches:
                citations[snippet_id] = snippet_content.strip()
            
            result["extraction_success"] = bool(answer)
            return context, answer, citations


def compute_combined_rubric_citation_reward(
    prediction: str, 
    label: str, 
    query: str,
    rubric_weight: float = 0.6,
    citation_weight: float = 0.4
) -> Dict[str, Any]:
    """
    Combined reward function that integrates rubrics and citation rewards with fine-grained token-level scoring.
    
    Args:
        prediction: The model's response text
        label: Ground truth or reference (should be JSON string with rubric information)
        query: The original query/question
        rubric_weight: Weight for rubric-based scoring (default: 0.6)
        citation_weight: Weight for citation-based scoring (default: 0.4)
    
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - rubric_reward: Individual rubric reward score
            - citation_reward: Individual citation reward score
            - log_values: Dict of metrics for logging
            - error: Error message if any component fails
    """
    
    # Validate weights
    if abs(rubric_weight + citation_weight - 1.0) > 1e-6:
        return {
            "finegrained_scores": [],
            "rubric_reward": 0.0,
            "citation_reward": 0.0,
            "log_values": {},
            "error": f"Weights must sum to 1.0, got {rubric_weight + citation_weight}"
        }
    
    result = {
        "finegrained_scores": [],
        "rubric_reward": 0.0,
        "citation_reward": 0.0,
        "log_values": {},
        "error": None
    }
    
    # Parse label if it's JSON, otherwise use as string
    try:
        if isinstance(label, str):
            parsed_label = json.loads(label)
        else:
            parsed_label = label
    except (json.JSONDecodeError, TypeError):
        result["error"] = "Failed to parse label as JSON for rubric evaluation"
        return result
    
    # Add query to parsed_label if not present
    if "Question" not in parsed_label and "query" not in parsed_label:
        parsed_label["Question"] = query
    
    # Extract answer and citations from the response
    try:
        extraction_result = {}
        extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(
            prediction, extraction_result
        )
        
        if not extracted_answer:
            result["error"] = "Failed to extract answer from response"
            return result
            
        result["log_values"]["citation_extraction_success"] = extraction_result.get("extraction_success", False)
        
    except Exception as e:
        result["error"] = f"Answer extraction failed: {str(e)}"
        return result
    
    # Compute rubric reward
    try:
        rubric_result = compute_rubric_reward(prediction, parsed_label)
        result["rubric_reward"] = rubric_result.get("reward", 0.0)
        result["rubric_scores"] = rubric_result.get("rubric_scores", {})
        
        # Copy rubric log values
        rubric_log_values = rubric_result.get("log_values", {})
        for key, value in rubric_log_values.items():
            result["log_values"][f"rubric_{key}"] = value
            
        if rubric_result.get("error"):
            result["error"] = f"Rubric error: {rubric_result['error']}"
            
    except Exception as e:
        result["error"] = f"Rubric computation failed: {str(e)}"
        result["rubric_reward"] = 0.0
    
    # Compute citation reward
    try:
        if extracted_citations and extracted_answer:
            citation_score = score_in_context_citations(query, extracted_answer, extracted_citations)
            result["citation_reward"] = citation_score
            result["log_values"]["citation_score"] = citation_score
            result["log_values"]["num_citations"] = len(extracted_citations)
        else:
            result["citation_reward"] = 0.0
            result["log_values"]["citation_score"] = 0.0
            result["log_values"]["num_citations"] = 0
            
    except Exception as e:
        if not result["error"]:
            result["error"] = f"Citation computation failed: {str(e)}"
        else:
            result["error"] += f"; Citation computation failed: {str(e)}"
        result["citation_reward"] = 0.0
    
    # Create fine-grained scores by parsing citation tokens in the answer
    try:
        finegrained_scores = []
        
        # Find the answer section in the original prediction to get correct character positions
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
        if not answer_match:
            # If no answer tags, use the entire prediction
            answer_start_in_prediction = 0
            answer_text = prediction
        else:
            answer_start_in_prediction = answer_match.start(1)  # Start of content inside <answer> tags
            answer_text = answer_match.group(1)
        
        # Find all citation spans within the answer
        cite_pattern = r'<cite id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</cite>'
        cite_matches = list(re.finditer(cite_pattern, answer_text, re.DOTALL))
        
        # Create spans for citation and non-citation tokens
        current_pos = 0
        response_idx = 0  # All spans belong to the same response
        
        for match in cite_matches:
            # Add non-citation text before this citation (if any)
            if match.start() > current_pos:
                non_cite_text = answer_text[current_pos:match.start()].strip()
                if non_cite_text:
                    start_char = answer_start_in_prediction + current_pos
                    end_char = answer_start_in_prediction + match.start()
                    finegrained_scores.append((
                        result["rubric_reward"],  # Use rubric score for non-citation tokens
                        (start_char, end_char),
                        0,  # Group 0 for rubric/content tokens
                        response_idx
                    ))
            
            # Add citation text with citation score
            cite_text = match.group(3).strip()
            if cite_text:
                start_char = answer_start_in_prediction + match.start(3)  # Start of cited content
                end_char = answer_start_in_prediction + match.end(3)      # End of cited content
                finegrained_scores.append((
                    result["citation_reward"],  # Use citation score for citation tokens
                    (start_char, end_char),
                    1,  # Group 1 for citation tokens
                    response_idx
                ))
            
            current_pos = match.end()
        
        # Add any remaining non-citation text after the last citation
        if current_pos < len(answer_text):
            remaining_text = answer_text[current_pos:].strip()
            if remaining_text:
                start_char = answer_start_in_prediction + current_pos
                end_char = answer_start_in_prediction + len(answer_text)
                finegrained_scores.append((
                    result["rubric_reward"],  # Use rubric score for non-citation tokens
                    (start_char, end_char),
                    0,  # Group 0 for rubric/content tokens
                    response_idx
                ))
        
        # If no citations found, assign rubric score to the entire answer
        if not cite_matches and answer_text.strip():
            start_char = answer_start_in_prediction
            end_char = answer_start_in_prediction + len(answer_text)
            finegrained_scores.append((
                result["rubric_reward"],
                (start_char, end_char),
                0,  # Group 0 for rubric/content tokens
                response_idx
            ))
        
        result["finegrained_scores"] = finegrained_scores
        
    except Exception as e:
        if not result["error"]:
            result["error"] = f"Fine-grained scoring failed: {str(e)}"
        else:
            result["error"] += f"; Fine-grained scoring failed: {str(e)}"
        result["finegrained_scores"] = []
    
    # Add weight information and summary stats to log values
    result["log_values"]["rubric_weight"] = rubric_weight
    result["log_values"]["citation_weight"] = citation_weight
    result["log_values"]["num_citation_spans"] = len([s for s in result["finegrained_scores"] if s[2] == 1])
    result["log_values"]["num_rubric_spans"] = len([s for s in result["finegrained_scores"] if s[2] == 0])
    result["log_values"]["total_spans"] = len(result["finegrained_scores"])
    
    # Calculate overall reward as weighted average (for logging purposes)
    if result["finegrained_scores"]:
        total_chars = sum(span[1][1] - span[1][0] for span in result["finegrained_scores"])
        if total_chars > 0:
            weighted_reward = sum(
                span[0] * (span[1][1] - span[1][0]) / total_chars 
                for span in result["finegrained_scores"]
            )
            result["log_values"]["overall_weighted_reward"] = weighted_reward
    
    return result


def compute_toy_reward_simple(prediction: str, label: str, query: str) -> Dict[str, Any]:
    """
    Simplified toy reward function for basic testing.
    Returns a simple score based on prediction length and keyword matching.
    
    Args:
        prediction: The model's response text
        label: Ground truth or reference (not used in this simple version)
        query: The original query/question
    
    Returns:
        Dict with reward score and basic metrics
    """
    
    # Basic length scoring (prefer responses between 100-500 characters)
    length = len(prediction)
    if length == 0:
        length_score = 0.0
    elif 100 <= length <= 500:
        length_score = 1.0
    else:
        # Penalty for being too short or too long
        if length < 100:
            length_score = length / 100.0
        else:
            length_score = max(0.1, 1.0 - (length - 500) / 1000.0)
    
    # Keyword matching score (simple heuristic)
    query_words = set(query.lower().split())
    prediction_words = set(prediction.lower().split())
    
    if len(query_words) > 0:
        keyword_overlap = len(query_words.intersection(prediction_words)) / len(query_words)
    else:
        keyword_overlap = 0.0
    
    # Check for answer-like patterns
    answer_patterns = ['answer', 'result', 'conclusion', 'therefore', 'because', 'due to']
    answer_score = 0.0
    for pattern in answer_patterns:
        if pattern in prediction.lower():
            answer_score += 0.2
    answer_score = min(1.0, answer_score)
    
    # Combine scores
    reward = 0.4 * length_score + 0.4 * keyword_overlap + 0.2 * answer_score
    
    return {
        "reward": reward,
        "log_values": {
            "length_score": length_score,
            "keyword_overlap": keyword_overlap,
            "answer_score": answer_score,
            "prediction_length": length,
            "query_length": len(query)
        }
    }


if __name__ == "__main__":
    print(f"Using {'real' if USING_REAL_FUNCTIONS else 'stub'} reward functions")
    print("="*60)
    
    # Test the combined reward function
    test_prediction = """
    <context>
    <search>Brazilian beachwear internationalization</search>
    <search>country of origin image Brazil fashion</search>
    <snippets id="115407351">Sutter's studies (2012) shows that design, quality and image are fundamental attributes in this market. The so-called 'Brasilidade', based on the valorization of the national culture, demonstrates that aspects that refer to attributes of Brazilian identity become a differential in the market.</snippets>
    </context>
    
    <answer>
    The Country of Origin Image (COI) significantly contributes to the internationalization and survival of Brazilian beachwear companies through several key mechanisms. 
    
    <cite id="115407351">Brazilian identity, known as 'Brasilidade', serves as a crucial product differentiator in foreign markets, with studies showing that design, quality and image are fundamental attributes that make buyers willing to pay more for products with Brazilian characteristics.</cite>
    
    This differentiation strategy leverages Brazil's unique cultural attributes including lifestyle elements, vibrant colors, national symbols, and natural raw materials that are strongly associated with the Brazilian fashion and beachwear sector.
    </answer>
    """
    
    test_label = {
        "Question": "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?",
        "Answer Critical": [
            {
                "Ingredient": "Explain how attributes related to the country of origin (i.e., Brazilian identity, 'Brasilidade') serve as product differentiators in foreign markets and positively affect buyer perception and willingness to pay.",
                "Handle": "Role of COI as Product Differentiator"
            }
        ]
    }
    
    test_query = "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?"
    
    # Test combined reward
    print("Testing combined rubric and citation reward:")
    result = compute_combined_rubric_citation_reward(
        test_prediction, 
        json.dumps(test_label), 
        test_query
    )
    
    print(f"Rubric reward: {result['rubric_reward']:.3f}")
    print(f"Citation reward: {result['citation_reward']:.3f}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    print(f"\nFine-grained scores ({len(result['finegrained_scores'])} spans):")
    for i, (score, (start, end), group_id, response_idx) in enumerate(result['finegrained_scores']):
        group_name = "Citation" if group_id == 1 else "Rubric"
        span_text = test_prediction[start:end]
        # Truncate long spans for display
        if len(span_text) > 100:
            span_text = span_text[:97] + "..."
        print(f"  {i+1}. {group_name} span [{start}:{end}] score={score:.3f}")
        print(f"     Text: {repr(span_text)}")
    
    print(f"\nLog values: {json.dumps(result['log_values'], indent=2)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test simple reward
    print("Testing simple toy reward:")
    simple_result = compute_toy_reward_simple(
        "The Brazilian beachwear industry benefits from country of origin image through cultural differentiation and brand recognition in international markets.",
        "",
        test_query
    )
    
    print(f"Simple reward: {simple_result['reward']:.3f}")
    print(f"Log values: {json.dumps(simple_result['log_values'], indent=2)}")
