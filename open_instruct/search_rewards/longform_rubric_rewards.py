import os
from typing import Any, Dict, Optional, Tuple
import json
import asyncio
import hashlib
import logging
from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations, compute_format_reward
from open_instruct.search_rewards.utils.rubric_utils import _score_rubric
from open_instruct.search_rewards.utils.search_utils import score_num_in_context_search_turns
from open_instruct.search_rewards.utils.citation_utils import score_in_context_citations

LOGGER = logging.getLogger(__name__)

# Reward weights for combining different reward components
REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}
REWARD_WEIGHTS_WITHOUT_CITATION = {
    "rubric_reward": 0.6,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.2,
}


def create_rubric_key(question: str, rubric: dict) -> str:
    """Create a unique key for a question-rubric pair to avoid title collisions."""
    # Combine question and rubric description for uniqueness
    content = f"{question}||{rubric['description']}||{rubric.get('title', '')}"
    # Create a short hash for efficiency
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return hash_obj.hexdigest()[:12]  # Use first 12 chars for readability


"""
Example test_case: 

ex = {"Question": "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?", "Answer Critical": [{"Ingredient": "Explain how attributes related to the country of origin (i.e., Brazilian identity, 'Brasilidade') serve as product differentiators in foreign markets and positively affect buyer perception and willingness to pay.", "Handle": "Role of COI as Product Differentiator", "Specifics": [{"Text": "Sutter's studies ( 2012) also shows that design, quality and image are fundamental attributes in this market...Sutter et. al (2014) also analyze the issue of the Brazilian identity as a product differentiating element. The so-called 'Brasilidade', based on the valorization of the national culture according to the study, demonstrates that aspects that refer to attributes of Brazilian identity become a differential in the market, interfering in the perception of the buyers that identify in this aspect a facet that would make them pay more for the product...In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}, {"Ingredient": "Describe how Brazilian lifestyle, colors, national symbols, raw materials, craft techniques, shapes, and sensuality are leveraged by companies as aspects of Brazil\u2019s country image to enhance appeal in the fashion and beachwear sector abroad.", "Handle": "Brazilian Attributes Leveraged for Appeal", "Specifics": [{"Text": "Sutter, Polo and Maclennan, (2014) also sought to understand the effects of Brazilianness into fashion. For such, authors found the following elements as particular of Brazil's image into fashion: life style, colors, representations of Brazil and national symbols, natural raw materials, fabrics, applications and craft techniques, shapes and volumes.", "Citation": "167755861"}, {"Text": "Guimar\u00e3es, Almeida, and Oliveira (2007), by analyzing the case of the jewelry brand H. Stern, exemplify how the company explores sensuality and beauty, which is another feature of the Brazilianness, in their product line.", "Citation": "167755861"}]}, {"Ingredient": "Mention how Country of Origin Image supports the internationalization process by enhancing product differentiation, motivating international expansion, and providing competitive advantage for Brazilian beachwear companies.", "Handle": "COI's Support for Internationalization", "Specifics": [{"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}, {"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector.", "Citation": "56041853"}]}], "Valuable": [{"Ingredient": "Discuss the influence of Brazil\u2019s country image as a brand and its exploitation in international promotion, including government and agency strategies (e.g., Apex, Brand Brazil).", "Handle": "COI as National Brand and Promotion", "Specifics": [{"Text": "Khauaja and Hemzo (2008) analyzed the construction of the Brand Brazil and and how it has been exploited by the Brazilian government agency Apex (Brazil's Export Promotion) in order to support the insertion of domestic products abroad.", "Citation": "167755861"}]}, {"Ingredient": "Describe how Brazilian beachwear products can serve diaspora and niche markets abroad, emphasizing potential for minimal adaptation due to their strong identification with Brazil.", "Handle": "Diaspora and Niche Markets for Beachwear", "Specifics": [{"Text": "This interpretation acknowledged two possibilities. On the one hand, it posited the existence of Brazilian niche markets abroad where products can be sold with the same characteristics as they are in the domestic market. One example mentioned was the case of Brazilian beachwear, a product strongly identified with the image of the country. Another considered the possibility of selling to Brazilians living abroad, or to segments where Brazilian products are in demand.", "Citation": "2388271"}]}, {"Ingredient": "Specify that internationalized Brazilian companies face challenges such as cultural differences and local adaptation but that maintaining a strong country image aids recognition and market entry.", "Handle": "Challenges and Recognition in Internationalization", "Specifics": [{"Text": "The main difficulties pointed out by internationalized Brazilian franchises are lack of knowledge of foreign markets, cultural differences, operational and legal difficulties and mainly the selection of suitable local franchisees...An investigation of the motivations for the internationalization of Brazilian franchises revealed the predominance of behavioral motivations, such as strengthening the company's image...", "Citation": "233775078"}, {"Text": "The impact of institutional image is the photography, reputation and stereotype that stakeholders link to companies of a specific country that carry domestic heritage by the influence of national background, mainly institutional characteristics (Kostova & Zaheer, 1999).", "Citation": "225598312"}]}], "Context": [{"Ingredient": "Include background on the early stage of research on Brazil\u2019s country of origin image and call for more empirical investigations in this field.", "Handle": "Early Stage of COI Research", "Specifics": [{"Text": "However, studies about COI which have Brazil as the object of research are still at an early stage (Giraldi, Giraldi, & Scaduto, 2011) claiming for more empirical investigations.", "Citation": "167755861"}, {"Text": "In addition, country image, according to Mariutti and Giraldi (2012), is an area of study that is gaining interest among researchers internationally.", "Citation": "203484639"}]}, {"Ingredient": "Describe how internationalization in the Brazilian beachwear sector requires strategic planning, investment, and is affected by the structural characteristics of Brazilian apparel firms.", "Handle": "Structural and Strategic Factors in Internationalization", "Specifics": [{"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector. However, internationalization is not an overnight change and requires time, investments and long run strategic planning, which hardly fits with the medium, small and micro enterprises constituting the very fabric of clothing industry in Brazil.", "Citation": "56041853"}, {"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}]}
messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["Question"]},
            ]
ground_truth = json.dumps(ex)
test_case = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
"""


def compute_rubric_reward(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False, mcp_parser_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute a reward score for a response based on a test case.
    """
    result = {
        "reward": 0.0,
        "rubric_averaged_reward": 0.0,
        "rubric_scores": {},
        "answer_extracted": None,
        "extraction_success": False,
        "error": None,
    }

    # Extract answer from the response
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    
    # Reward number of search turns
    in_context_search_turns_reward, num_search_turns = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)

    if extracted_answer is None:
        result["error"] = "Failed to extract answer from response"
        result["reward"] = 0.0
        result["log_values"] = {
            "format_correct_has_answer": 0.0,
            "num_search_turns": num_search_turns,
            "in_context_search_turns_reward": in_context_search_turns_reward,
            "rubric_averaged_reward": 0.0,
        }
        return result

    # Score the rubrics using the extracted function
    rubric_scores = _score_rubric(extracted_answer, ground_truth, use_general_rubric=use_general_rubric)
                    
    result["rubric_scores"] = rubric_scores
    result["rubric_averaged_reward"] = sum(rubric_scores.values()) / len(rubric_scores)
    result["reward"] = 0.8 * result["rubric_averaged_reward"] \
        + 0.2 * in_context_search_turns_reward
    result["log_values"] = {
        "format_correct_has_answer": 1.0,
        "in_context_search_turns_reward": in_context_search_turns_reward,
        "num_search_turns": num_search_turns,
        "rubric_averaged_reward": result["rubric_averaged_reward"],
        }
    return result


def compute_weighted_rubric_reward(response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a reward score for a response based on a test case.
    """
    num_rubrics = len(ground_truth["rubrics"])
    print("🔥 Num rubrics: ", num_rubrics)
    
    result = {
        "reward": 0.0,
        "log_values": {
            "format_correct_has_answer": 0.0,
            "rubric_reward": 0.0,
            "num_rubrics": num_rubrics,
        },
        "error": None,
    }

    # Extract answer from the response
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    
    if extracted_answer is None:
        result["error"] = "Failed to extract answer from response"
        result["reward"] = 0.0
        
        # Create zero scores for all rubrics in the ground truth
        rubrics = ground_truth["rubrics"]
        question = ground_truth["query"]
        rubric_scores_by_title = {}
        for rubric in rubrics:
            rubric_key = create_rubric_key(question, rubric)
            rubric_scores_by_title[rubric_key] = 0.0
        
        result["log_values"] = {
            "rubric_reward": 0.0,
            "format_correct_has_answer": 0.0,
            "num_rubrics": len(rubrics),
            "rubric_scores_by_title": rubric_scores_by_title,
        }
        return result

    # Compute per-rubric scores grouped by title (this includes all the expensive scoring)
    rubric_scores_by_title, overall_reward = asyncio.run(_compute_rubric_scores_and_reward(extracted_answer, ground_truth))
                    
    result["reward"] = overall_reward
    result["log_values"] = {
        "rubric_reward": overall_reward,
        "format_correct_has_answer": 1.0,
        "num_rubrics": num_rubrics,
        "rubric_scores_by_title": rubric_scores_by_title,
        }
    return result


def compute_weighted_rubric_reward_with_citation_and_format_reward(response: str, ground_truth: Dict[str, Any], mcp_parser_name: Optional[str] = None, use_general_rubric: bool = False, no_citation_reward: bool = False, use_likert_rubric: bool = False) -> Dict[str, Any]:
    """
    Compute a comprehensive reward score that includes rubric, citation, format, and search turn rewards.
    
    This function uses the current module's rubric computation logic and adds citation, format, 
    and search turn rewards copied from the longform_averaged_outcome_rewards module.
    
    Args:
        response: The response text to score
        ground_truth: Dictionary containing the question and other ground truth data
        mcp_parser_name: Optional parser name for format scoring
        use_general_rubric: Whether to use general rubric scoring
        
    Returns:
        Dictionary containing reward components and overall reward
    """
    num_rubrics = len(ground_truth["rubrics"])
    question = ground_truth.get("query") or ground_truth.get("Question", "")
    
    # Extract answer, context, and citations from response
    extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    
    # Initialize result structure
    result = {
        "reward": 0.0,
        "log_values": {
            "format_correct_has_answer": 0.0,
            "rubric_reward": 0.0,
            "citation_reward": 0.0,
            "format_reward": 0.0,
            "num_search_turns_reward": 0.0,
            "num_rubrics": num_rubrics,
        },
        "error": None,
    }
    
    # Score format reward (copied from longform_averaged_outcome_rewards)
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)
    result["log_values"]["format_reward"] = format_reward
    
    # Score num search turns reward (copied from longform_averaged_outcome_rewards)
    num_search_turns_reward, num_search_turns = score_num_in_context_search_turns(extracted_context, mcp_parser_name=mcp_parser_name)
    result["log_values"]["num_search_turns_reward"] = num_search_turns_reward
    
    if extracted_answer is None:
        result["error"] = "Failed to extract answer from response"
        result["reward"] = 0.0
        
        # Create zero scores for all rubrics in the ground truth
        rubrics = ground_truth["rubrics"]
        rubric_scores_by_title = {}
        for rubric in rubrics:
            rubric_key = create_rubric_key(question, rubric)
            rubric_scores_by_title[rubric_key] = 0.0
        
        result["log_values"]["rubric_scores_by_title"] = rubric_scores_by_title
        
        # Compute final reward using only format and search turn rewards
        reward = 0.0
        if no_citation_reward:
            weights = REWARD_WEIGHTS_WITHOUT_CITATION
        else:
            weights = REWARD_WEIGHTS
        for key, weight in weights.items():
            if key in result["log_values"]:
                reward += weight * result["log_values"][key]
        result["reward"] = reward
        
        return result

    # Compute per-rubric scores grouped by title using existing logic
    rubric_scores_by_title, rubric_reward = asyncio.run(_compute_rubric_scores_and_reward(extracted_answer, ground_truth, use_general_rubric=use_general_rubric, use_likert_rubric=use_likert_rubric))
    result["log_values"]["rubric_reward"] = rubric_reward
    result["log_values"]["rubric_scores_by_title"] = rubric_scores_by_title
    result["log_values"]["format_correct_has_answer"] = 1.0
    
    # Score citation reward (copied from longform_averaged_outcome_rewards)
    if not no_citation_reward:
        citation_reward = score_in_context_citations(question, response, extracted_citations)
    else:
        citation_reward = 0.0
    result["log_values"]["citation_reward"] = citation_reward
    
    # Compute final weighted reward
    reward = 0.0
    if no_citation_reward:
        weights = REWARD_WEIGHTS_WITHOUT_CITATION
    else:
        weights = REWARD_WEIGHTS
    for key, weight in weights.items():
        if key in result["log_values"]:
            reward += weight * result["log_values"][key]
    result["reward"] = reward
    
    return result


async def _compute_rubric_scores_and_reward(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False, use_likert_rubric: bool = False) -> Tuple[Dict[str, float], float]:
    """
    Compute both per-rubric scores grouped by title AND the overall weighted reward in a single pass.
    
    Args:
        response: The response text to score
        ground_truth: Dictionary containing rubrics and query
        
    Returns:
        Tuple of (title_scores_dict, overall_weighted_reward)
    """
    from open_instruct.search_rewards.utils.rubric_utils import _score_property_async
    
    rubrics = ground_truth["rubrics"]
    question = ground_truth["query"]
    
    # Group rubrics by title and compute scores in one pass
    title_groups = {}
    tasks = []
    task_to_rubric_mapping = []
    
    
    if use_likert_rubric:
        system_prompt = """You are an expert evaluator. Given a user prompt and a generated response, please rate the overall quality of the response on a scale of 1 to 10, where 1 is very poor and 10 is excellent.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain a single key "score" and the value should be an integer between 1 and 10.
Example response:
```json
{
"score": 8
}```"""
        user_prompt = f"""Given the following prompt, and response, please rate the overall quality of the response on a scale of 1 to 10.
<prompt>
{question}
</prompt>   
<response>
{response}
</response>
Your JSON Evaluation:"""
        task = _score_property_async(response, question, None, system_prompt=system_prompt, user_prompt=user_prompt, score_scale=10.0)
        tasks.append(task)
        task_to_rubric_mapping.append(("likert_rubric", system_prompt))
    elif use_general_rubric:
            general_rubric = """(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible
                (2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially
                (3) Factuality: There should be minimal factual errors
                (4) Coherence: The discussion should stay focused and relevant to the topic"""
            task = _score_property_async(response, question, general_rubric)
            tasks.append(task)
            task_to_rubric_mapping.append(("general_rubric", general_rubric))
    else:
        for rubric in rubrics:
            rubric_key = create_rubric_key(question, rubric)
            
            if rubric_key not in title_groups:
                title_groups[rubric_key] = {"scores": [], "weights": []}
            
            # Create async task for scoring this rubric   
            task = _score_property_async(response, question, rubric["description"])
            tasks.append(task)
            task_to_rubric_mapping.append((rubric_key, rubric))
    
    # Execute all scoring tasks in parallel (this is the expensive part, done only once)
    scores = await asyncio.gather(*tasks)
    
    if use_likert_rubric:
        title_scores = {"likert_rubric": scores[0]}
        return title_scores, scores[0]
    elif use_general_rubric:
        title_scores = {"general_rubric": scores[0]}
        return title_scores, scores[0]
    
    # Organize results by title and compute overall reward simultaneously
    title_scores = {}
    overall_weighted_sum = 0.0
    overall_total_weight = 0.0
    
    for score, (rubric_key, rubric) in zip(scores, task_to_rubric_mapping):
        # Add to title groups
        title_groups[rubric_key]["scores"].append(score)
        title_groups[rubric_key]["weights"].append(rubric["weight"])
        
        # Add to overall weighted sum
        overall_weighted_sum += score * rubric["weight"]
        if rubric["weight"] > 0:
            overall_total_weight += rubric["weight"]
    
    # Compute weighted average for each rubric key group
    for rubric_key, group_data in title_groups.items():
        scores_list = group_data["scores"]
        weights_list = group_data["weights"]
        
        if scores_list and weights_list:
            # Compute weighted average
            weighted_sum = sum(s * w for s, w in zip(scores_list, weights_list))
            total_weight = sum(w for w in weights_list if w > 0)
            title_scores[rubric_key] = weighted_sum / max(total_weight, 1.0)
        else:
            title_scores[rubric_key] = 0.0
    
    # Compute overall weighted reward
    overall_reward = overall_weighted_sum / max(overall_total_weight, 1.0)
    
    return title_scores, overall_reward


if __name__ == '__main__':
    # Example usage:
    ground_truth = {"Question": "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?", "Answer Critical": [{"Ingredient": "Explain how attributes related to the country of origin (i.e., Brazilian identity, 'Brasilidade') serve as product differentiators in foreign markets and positively affect buyer perception and willingness to pay.", "Handle": "Role of COI as Product Differentiator", "Specifics": [{"Text": "Sutter's studies ( 2012) also shows that design, quality and image are fundamental attributes in this market...Sutter et. al (2014) also analyze the issue of the Brazilian identity as a product differentiating element. The so-called 'Brasilidade', based on the valorization of the national culture according to the study, demonstrates that aspects that refer to attributes of Brazilian identity become a differential in the market, interfering in the perception of the buyers that identify in this aspect a facet that would make them pay more for the product...In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}, {"Ingredient": "Describe how Brazilian lifestyle, colors, national symbols, raw materials, craft techniques, shapes, and sensuality are leveraged by companies as aspects of Brazil\u2019s country image to enhance appeal in the fashion and beachwear sector abroad.", "Handle": "Brazilian Attributes Leveraged for Appeal", "Specifics": [{"Text": "Sutter, Polo and Maclennan, (2014) also sought to understand the effects of Brazilianness into fashion. For such, authors found the following elements as particular of Brazil's image into fashion: life style, colors, representations of Brazil and national symbols, natural raw materials, fabrics, applications and craft techniques, shapes and volumes.", "Citation": "167755861"}, {"Text": "Guimar\u00e3es, Almeida, and Oliveira (2007), by analyzing the case of the jewelry brand H. Stern, exemplify how the company explores sensuality and beauty, which is another feature of the Brazilianness, in their product line.", "Citation": "167755861"}]}, {"Ingredient": "Mention how Country of Origin Image supports the internationalization process by enhancing product differentiation, motivating international expansion, and providing competitive advantage for Brazilian beachwear companies.", "Handle": "COI's Support for Internationalization", "Specifics": [{"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}, {"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector.", "Citation": "56041853"}]}], "Valuable": [{"Ingredient": "Discuss the influence of Brazil\u2019s country image as a brand and its exploitation in international promotion, including government and agency strategies (e.g., Apex, Brand Brazil).", "Handle": "COI as National Brand and Promotion", "Specifics": [{"Text": "Khauaja and Hemzo (2008) analyzed the construction of the Brand Brazil and and how it has been exploited by the Brazilian government agency Apex (Brazil's Export Promotion) in order to support the insertion of domestic products abroad.", "Citation": "167755861"}]}, {"Ingredient": "Describe how Brazilian beachwear products can serve diaspora and niche markets abroad, emphasizing potential for minimal adaptation due to their strong identification with Brazil.", "Handle": "Diaspora and Niche Markets for Beachwear", "Specifics": [{"Text": "This interpretation acknowledged two possibilities. On the one hand, it posited the existence of Brazilian niche markets abroad where products can be sold with the same characteristics as they are in the domestic market. One example mentioned was the case of Brazilian beachwear, a product strongly identified with the image of the country. Another considered the possibility of selling to Brazilians living abroad, or to segments where Brazilian products are in demand.", "Citation": "2388271"}]}, {"Ingredient": "Specify that internationalized Brazilian companies face challenges such as cultural differences and local adaptation but that maintaining a strong country image aids recognition and market entry.", "Handle": "Challenges and Recognition in Internationalization", "Specifics": [{"Text": "The main difficulties pointed out by internationalized Brazilian franchises are lack of knowledge of foreign markets, cultural differences, operational and legal difficulties and mainly the selection of suitable local franchisees...An investigation of the motivations for the internationalization of Brazilian franchises revealed the predominance of behavioral motivations, such as strengthening the company's image...", "Citation": "233775078"}, {"Text": "The impact of institutional image is the photography, reputation and stereotype that stakeholders link to companies of a specific country that carry domestic heritage by the influence of national background, mainly institutional characteristics (Kostova & Zaheer, 1999).", "Citation": "225598312"}]}], "Context": [{"Ingredient": "Include background on the early stage of research on Brazil\u2019s country of origin image and call for more empirical investigations in this field.", "Handle": "Early Stage of COI Research", "Specifics": [{"Text": "However, studies about COI which have Brazil as the object of research are still at an early stage (Giraldi, Giraldi, & Scaduto, 2011) claiming for more empirical investigations.", "Citation": "167755861"}, {"Text": "In addition, country image, according to Mariutti and Giraldi (2012), is an area of study that is gaining interest among researchers internationally.", "Citation": "203484639"}]}, {"Ingredient": "Describe how internationalization in the Brazilian beachwear sector requires strategic planning, investment, and is affected by the structural characteristics of Brazilian apparel firms.", "Handle": "Structural and Strategic Factors in Internationalization", "Specifics": [{"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector. However, internationalization is not an overnight change and requires time, investments and long run strategic planning, which hardly fits with the medium, small and micro enterprises constituting the very fabric of clothing industry in Brazil.", "Citation": "56041853"}, {"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}]}
    
    response = "<context>Some context here. <search>First search query</search> <search>Second search query</search></context><answer>The Country of Origin Image (COI) significantly impacts the internationalization and survival of Brazilian beachwear companies by leveraging 'Brasilidade' as a key product differentiator. This includes using Brazilian lifestyle elements like colors, national symbols, and natural raw materials. The COI supports international expansion by creating a competitive advantage.</answer>"
    response = "<context>Some context here. <tool name='google_search'>First search query</tool> <tool name='google_search'>Second search query</tool></context><answer>The Country of Origin Image (COI) significantly impacts the internationalization and survival of Brazilian beachwear companies by leveraging 'Brasilidade' as a key product differentiator. This includes using Brazilian lifestyle elements like colors, national symbols, and natural raw materials. The COI supports international expansion by creating a competitive advantage.</answer>"
    
    
    rewards = compute_rubric_reward(response, ground_truth, use_general_rubric=True, mcp_parser_name="unified")
    
    if rewards.get("error"):
        print(f"An error occurred: {rewards['error']}")

    print(json.dumps(rewards, indent=2))