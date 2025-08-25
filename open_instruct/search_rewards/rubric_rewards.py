import os
from typing import Any, Dict
import json
from open_instruct.search_rewards.format_utils import extract_answer_context_citations
from open_instruct.search_rewards.run_utils import extract_json_from_response, run_litellm
from open_instruct.search_rewards.citation_rewards_utils import score_num_in_context_search_turns



"""
Example test_case: 

ex = {"Question": "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?", "Answer Critical": [{"Ingredient": "Explain how attributes related to the country of origin (i.e., Brazilian identity, 'Brasilidade') serve as product differentiators in foreign markets and positively affect buyer perception and willingness to pay.", "Handle": "Role of COI as Product Differentiator", "Specifics": [{"Text": "Sutter's studies ( 2012) also shows that design, quality and image are fundamental attributes in this market...Sutter et. al (2014) also analyze the issue of the Brazilian identity as a product differentiating element. The so-called 'Brasilidade', based on the valorization of the national culture according to the study, demonstrates that aspects that refer to attributes of Brazilian identity become a differential in the market, interfering in the perception of the buyers that identify in this aspect a facet that would make them pay more for the product...In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}, {"Ingredient": "Describe how Brazilian lifestyle, colors, national symbols, raw materials, craft techniques, shapes, and sensuality are leveraged by companies as aspects of Brazil\u2019s country image to enhance appeal in the fashion and beachwear sector abroad.", "Handle": "Brazilian Attributes Leveraged for Appeal", "Specifics": [{"Text": "Sutter, Polo and Maclennan, (2014) also sought to understand the effects of Brazilianness into fashion. For such, authors found the following elements as particular of Brazil's image into fashion: life style, colors, representations of Brazil and national symbols, natural raw materials, fabrics, applications and craft techniques, shapes and volumes.", "Citation": "167755861"}, {"Text": "Guimar\u00e3es, Almeida, and Oliveira (2007), by analyzing the case of the jewelry brand H. Stern, exemplify how the company explores sensuality and beauty, which is another feature of the Brazilianness, in their product line.", "Citation": "167755861"}]}, {"Ingredient": "Mention how Country of Origin Image supports the internationalization process by enhancing product differentiation, motivating international expansion, and providing competitive advantage for Brazilian beachwear companies.", "Handle": "COI's Support for Internationalization", "Specifics": [{"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}, {"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector.", "Citation": "56041853"}]}], "Valuable": [{"Ingredient": "Discuss the influence of Brazil\u2019s country image as a brand and its exploitation in international promotion, including government and agency strategies (e.g., Apex, Brand Brazil).", "Handle": "COI as National Brand and Promotion", "Specifics": [{"Text": "Khauaja and Hemzo (2008) analyzed the construction of the Brand Brazil and and how it has been exploited by the Brazilian government agency Apex (Brazil's Export Promotion) in order to support the insertion of domestic products abroad.", "Citation": "167755861"}]}, {"Ingredient": "Describe how Brazilian beachwear products can serve diaspora and niche markets abroad, emphasizing potential for minimal adaptation due to their strong identification with Brazil.", "Handle": "Diaspora and Niche Markets for Beachwear", "Specifics": [{"Text": "This interpretation acknowledged two possibilities. On the one hand, it posited the existence of Brazilian niche markets abroad where products can be sold with the same characteristics as they are in the domestic market. One example mentioned was the case of Brazilian beachwear, a product strongly identified with the image of the country. Another considered the possibility of selling to Brazilians living abroad, or to segments where Brazilian products are in demand.", "Citation": "2388271"}]}, {"Ingredient": "Specify that internationalized Brazilian companies face challenges such as cultural differences and local adaptation but that maintaining a strong country image aids recognition and market entry.", "Handle": "Challenges and Recognition in Internationalization", "Specifics": [{"Text": "The main difficulties pointed out by internationalized Brazilian franchises are lack of knowledge of foreign markets, cultural differences, operational and legal difficulties and mainly the selection of suitable local franchisees...An investigation of the motivations for the internationalization of Brazilian franchises revealed the predominance of behavioral motivations, such as strengthening the company's image...", "Citation": "233775078"}, {"Text": "The impact of institutional image is the photography, reputation and stereotype that stakeholders link to companies of a specific country that carry domestic heritage by the influence of national background, mainly institutional characteristics (Kostova & Zaheer, 1999).", "Citation": "225598312"}]}], "Context": [{"Ingredient": "Include background on the early stage of research on Brazil\u2019s country of origin image and call for more empirical investigations in this field.", "Handle": "Early Stage of COI Research", "Specifics": [{"Text": "However, studies about COI which have Brazil as the object of research are still at an early stage (Giraldi, Giraldi, & Scaduto, 2011) claiming for more empirical investigations.", "Citation": "167755861"}, {"Text": "In addition, country image, according to Mariutti and Giraldi (2012), is an area of study that is gaining interest among researchers internationally.", "Citation": "203484639"}]}, {"Ingredient": "Describe how internationalization in the Brazilian beachwear sector requires strategic planning, investment, and is affected by the structural characteristics of Brazilian apparel firms.", "Handle": "Structural and Strategic Factors in Internationalization", "Specifics": [{"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector. However, internationalization is not an overnight change and requires time, investments and long run strategic planning, which hardly fits with the medium, small and micro enterprises constituting the very fabric of clothing industry in Brazil.", "Citation": "56041853"}, {"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}]}
messages = messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["Question"]},
            ]
ground_truth = json.dumps(ex)
test_case = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
"""

def _score_property(response: str, question: str, prop: str) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion
    on a scale of 0-2.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return: score between 0 and 1 after normalizing the LLM score
    """
    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}."""
    user_prompt = (
        f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
    )

    resp = run_litellm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
    )

    obj = extract_json_from_response(resp)
    if not obj:
        return 0.0

    return obj["score"] / 2.0


def _score_rubric(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False) -> Dict[str, float]:
    """
    Score the response against all rubrics in the ground truth.
    
    Args:
        response: The extracted answer text to be scored
        ground_truth: Dictionary containing the question and rubrics
        
    Returns:
        Dictionary mapping rubric handles to their scores (0.0 to 1.0)
    """
    question = ground_truth["Question"] if "Question" in ground_truth else ground_truth["query"]
    
    rubric_scores = {}
    
    if use_general_rubric:
        general_rubric = """(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible
(2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially
(3) Factuality: There should be minimal factual errors
(4) Coherence: The discussion should stay focused and relevant to the topic"""
        score = _score_property(response, question, general_rubric)
        rubric_scores["general"] = score
        return rubric_scores
    elif "rubric" in ground_truth:
        for rubric in ground_truth["rubric"]:
            handle = rubric.get("type")
            rubric_text = rubric["rubric_item"]
            score = _score_property(response, question, rubric_text)
            rubric_scores[handle] = score
    elif "Answer Critical" in ground_truth:
        for rubric in ground_truth["Answer Critical"]:
            handle = rubric.get("Handle")
            rubric_text = rubric["Ingredient"]
            score = _score_property(response, question, rubric_text)
            rubric_scores[handle] = score
    else:
        raise ValueError(f"Unsupported rubric format found in ground truth: {ground_truth}")
    
    return rubric_scores

def compute_rubric_reward(response: str, ground_truth: Dict[str, Any], use_general_rubric: bool = False) -> Dict[str, Any]:
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
    in_context_search_turns_reward, num_search_turns = score_num_in_context_search_turns(extracted_context)

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

if __name__ == '__main__':
    # Example usage:
    ground_truth = {"Question": "How does the Country of Origin Image contribute to the internationalization and survival of brazilian companies in the beachwear sector abroad?", "Answer Critical": [{"Ingredient": "Explain how attributes related to the country of origin (i.e., Brazilian identity, 'Brasilidade') serve as product differentiators in foreign markets and positively affect buyer perception and willingness to pay.", "Handle": "Role of COI as Product Differentiator", "Specifics": [{"Text": "Sutter's studies ( 2012) also shows that design, quality and image are fundamental attributes in this market...Sutter et. al (2014) also analyze the issue of the Brazilian identity as a product differentiating element. The so-called 'Brasilidade', based on the valorization of the national culture according to the study, demonstrates that aspects that refer to attributes of Brazilian identity become a differential in the market, interfering in the perception of the buyers that identify in this aspect a facet that would make them pay more for the product...In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}, {"Ingredient": "Describe how Brazilian lifestyle, colors, national symbols, raw materials, craft techniques, shapes, and sensuality are leveraged by companies as aspects of Brazil\u2019s country image to enhance appeal in the fashion and beachwear sector abroad.", "Handle": "Brazilian Attributes Leveraged for Appeal", "Specifics": [{"Text": "Sutter, Polo and Maclennan, (2014) also sought to understand the effects of Brazilianness into fashion. For such, authors found the following elements as particular of Brazil's image into fashion: life style, colors, representations of Brazil and national symbols, natural raw materials, fabrics, applications and craft techniques, shapes and volumes.", "Citation": "167755861"}, {"Text": "Guimar\u00e3es, Almeida, and Oliveira (2007), by analyzing the case of the jewelry brand H. Stern, exemplify how the company explores sensuality and beauty, which is another feature of the Brazilianness, in their product line.", "Citation": "167755861"}]}, {"Ingredient": "Mention how Country of Origin Image supports the internationalization process by enhancing product differentiation, motivating international expansion, and providing competitive advantage for Brazilian beachwear companies.", "Handle": "COI's Support for Internationalization", "Specifics": [{"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}, {"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector.", "Citation": "56041853"}]}], "Valuable": [{"Ingredient": "Discuss the influence of Brazil\u2019s country image as a brand and its exploitation in international promotion, including government and agency strategies (e.g., Apex, Brand Brazil).", "Handle": "COI as National Brand and Promotion", "Specifics": [{"Text": "Khauaja and Hemzo (2008) analyzed the construction of the Brand Brazil and and how it has been exploited by the Brazilian government agency Apex (Brazil's Export Promotion) in order to support the insertion of domestic products abroad.", "Citation": "167755861"}]}, {"Ingredient": "Describe how Brazilian beachwear products can serve diaspora and niche markets abroad, emphasizing potential for minimal adaptation due to their strong identification with Brazil.", "Handle": "Diaspora and Niche Markets for Beachwear", "Specifics": [{"Text": "This interpretation acknowledged two possibilities. On the one hand, it posited the existence of Brazilian niche markets abroad where products can be sold with the same characteristics as they are in the domestic market. One example mentioned was the case of Brazilian beachwear, a product strongly identified with the image of the country. Another considered the possibility of selling to Brazilians living abroad, or to segments where Brazilian products are in demand.", "Citation": "2388271"}]}, {"Ingredient": "Specify that internationalized Brazilian companies face challenges such as cultural differences and local adaptation but that maintaining a strong country image aids recognition and market entry.", "Handle": "Challenges and Recognition in Internationalization", "Specifics": [{"Text": "The main difficulties pointed out by internationalized Brazilian franchises are lack of knowledge of foreign markets, cultural differences, operational and legal difficulties and mainly the selection of suitable local franchisees...An investigation of the motivations for the internationalization of Brazilian franchises revealed the predominance of behavioral motivations, such as strengthening the company's image...", "Citation": "233775078"}, {"Text": "The impact of institutional image is the photography, reputation and stereotype that stakeholders link to companies of a specific country that carry domestic heritage by the influence of national background, mainly institutional characteristics (Kostova & Zaheer, 1999).", "Citation": "225598312"}]}], "Context": [{"Ingredient": "Include background on the early stage of research on Brazil\u2019s country of origin image and call for more empirical investigations in this field.", "Handle": "Early Stage of COI Research", "Specifics": [{"Text": "However, studies about COI which have Brazil as the object of research are still at an early stage (Giraldi, Giraldi, & Scaduto, 2011) claiming for more empirical investigations.", "Citation": "167755861"}, {"Text": "In addition, country image, according to Mariutti and Giraldi (2012), is an area of study that is gaining interest among researchers internationally.", "Citation": "203484639"}]}, {"Ingredient": "Describe how internationalization in the Brazilian beachwear sector requires strategic planning, investment, and is affected by the structural characteristics of Brazilian apparel firms.", "Handle": "Structural and Strategic Factors in Internationalization", "Specifics": [{"Text": "Segments like beachwear, lingerie and jeans-wear constitute the very strengths of the Brazilian apparel industry, worldwide recognised, and these product lines may drive the internationalization of the sector. However, internationalization is not an overnight change and requires time, investments and long run strategic planning, which hardly fits with the medium, small and micro enterprises constituting the very fabric of clothing industry in Brazil.", "Citation": "56041853"}, {"Text": "In order for this process to occur and the company to internationalize, some characteristics are identified by the study of (Sutter et al., 2016), they are: design, product quality, support and services, product attributes related to the country of origin and adequate price.", "Citation": "115407351"}]}]}
    
    response = "<context>Some context here. <search>First search query</search> <search>Second search query</search></context><answer>The Country of Origin Image (COI) significantly impacts the internationalization and survival of Brazilian beachwear companies by leveraging 'Brasilidade' as a key product differentiator. This includes using Brazilian lifestyle elements like colors, national symbols, and natural raw materials. The COI supports international expansion by creating a competitive advantage.</answer>"
    
    rewards = compute_rubric_reward(response, ground_truth, use_general_rubric=True)
    
    if rewards.get("error"):
        print(f"An error occurred: {rewards['error']}")

    print(json.dumps(rewards, indent=2))