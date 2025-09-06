import logging
import os
from typing import Any, Dict

from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_litellm

LOGGER = logging.getLogger(__name__)



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

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = run_litellm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
        )

        obj = extract_json_from_response(resp)
        if not obj:
            return 0.0
        
        # Validate that obj is a dictionary and has the expected structure
        if not isinstance(obj, dict) or "score" not in obj:
            LOGGER.warning(f"Invalid JSON structure in response: {obj}")
            return 0.0
            
        # Validate that score is a number
        try:
            score = float(obj["score"])
            return score / 2.0
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in response: {obj['score']}, error: {e}")
            return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0
    

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



def _score_property_with_spans(response: str, question: str, prop: str) -> float:
    """
    Score the response as per the annotation rubric/criterion represented here by ``prop``.
    The score is calculated by asking an LLM to judge the response for satisfaction of the rubric/criterion
    on a scale of 0-2. 
    In addition, output the spans of the response that are judged as satisfying the criterion.
    :param response: the response to be scored
    :param question: the question for which the response is being scored
    :param prop: the rubric/criterion to be satisfied
    :return in a json format: 
        score (int) between 0 and 1 after normalizing the LLM score
        spans (list of strings) of the verbatim response that are judged as satisfying the criterion
    """
    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion. Judge only the specified aspect(s), not any other qualities of the answer. In addition, output the spans of the response that are judged as satisfying the criterion. You should output the spans in the format of verbatim snippets from the response that are judged as satisfying the criterion. Output JSON in the format: {{"score": x, "spans": [y]}}."""
    user_prompt = (
        f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>"""
    )

    # wrap in try-except to handle litellm API errors
    # these might just be ephemeral, so we don't want to crash the whole training job.
    try:
        resp = run_litellm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1"),
        )

        obj = extract_json_from_response(resp)
        if not obj:
            LOGGER.warning(f"No JSON object found in rubric span tag response: {resp}")
            return 0.0, []
        
        # Validate that obj is a dictionary and has the expected structure
        if not isinstance(obj, dict) or "score" not in obj or "spans" not in obj:
            LOGGER.warning(f"Invalid JSON structure in rubric span tag response: {obj}")
            return 0.0, []
            
        # Validate that score is a number
        try:
            score = float(obj["score"])
            spans = obj["spans"]
            for span in spans:
                print("Span included in the response: ", span in response)
            print("Tagged response ratio: ", len(" ".join(spans)) / len(response))
            return score / 2.0, spans
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Invalid score value in rubric span tag response: {obj['score']}, error: {e}")
            return 0.0, []
            
    except Exception as e:
        LOGGER.warning(f"Error scoring rubric: {e}")
        return 0.0, []