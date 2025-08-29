import logging
import re
from typing import Any, Dict, List, Tuple, Optional, Union
import json
import string

from open_instruct.search_rewards.find_reward_spans import FinegrainedScore

LOGGER = logging.getLogger(__name__)


def split_response_and_get_spans(response: str, num_questions: int) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """
    Split a multi-question response into individual question responses and get their spans.
    Splits by </answer{i}> tags where i is the question number (1-indexed).
    
    Args:
        response: The full response containing multiple answers
        num_questions: Number of questions expected
        
    Returns:
        Tuple of (sub_responses, spans) where:
        - sub_responses: List of response text for each question
        - spans: List of span tuples [(start_char, end_char)] for each question
    """
    if num_questions <= 1:
        return [response], [[(0, len(response))]]
    
    sub_responses = []
    spans = []
    num_valid_spans = 0
    
    # Find split points using </answer{i}> tags
    start_char = 0
    for i in range(1, num_questions + 1):
        end_tag = f"</answer{i}>"
        match = re.search(re.escape(end_tag), response[start_char:])
        if match:
            end_char = start_char + match.end()
            sub_responses.append(response[start_char:end_char])
            spans.append([(start_char, end_char)])
            num_valid_spans += 1
            start_char = end_char
        elif i == num_questions:
            sub_responses.append(response[start_char:])
            spans.append([(start_char, len(response))])
            break
        else:
            # # Otherwise, return invalid spans that will trigger the fallback mechanism to use the entire response
            # sub_responses.append(response)
            # spans.append([(-1, -1)])
            # Or, penalize the entire response and return empty sub_response
            sub_responses.append("")
            spans.append([(-1, -1)])
        
    return sub_responses, spans, num_valid_spans


def extract_ground_truth_per_question(ground_truth: str) -> List[str]:
    """
    Extract ground truth answers for each individual question.
    Expects ground truth in format: JSON arrays separated by semicolons
    e.g., '["ans1", "ans2"]; ["ans3", "ans4"]; ["ans5"]'
    Returns: List of JSON strings, one per question
    """
    try:
        # Try to parse as a complete JSON array first for single question case
        if ground_truth.strip().startswith('[') and ground_truth.strip().endswith(']'):
            parsed = json.loads(ground_truth)
            return [json.dumps(item) if isinstance(item, list) else str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Handle the multi-question case: JSON arrays separated by semicolons
    try:
        if ground_truth.strip().startswith('[') and ground_truth.strip().endswith(']'):
            # Convert semicolon-separated JSON arrays to a proper JSON array
            full_json = '[' + ground_truth.replace('; ', ', ') + ']'
            parsed = json.loads(full_json)
            return [json.dumps(item) if isinstance(item, list) else str(item) for item in parsed]
        else:
            return ground_truth.split(";")
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Invalid ground truth format: {ground_truth}")


def extract_boxed_answer_from_response(response: str) -> str:
    """
    Extract the boxed answer from the response.
    Looks for content in \\boxed{...} format anywhere in the response.
    Returns the content inside the boxed expression, or empty string if not found.
    """
    # Look for \\boxed{...} pattern (flexible with whitespace)
    boxed_match = re.search(r"\\boxed\s*\{\s*(.*?)\s*\}", response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fallback: look for \boxed{...} (single backslash)
    boxed_match = re.search(r"\boxed\s*\{\s*(.*?)\s*\}", response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no boxed answer found, return the entire response
    return response

def normalize_answer(s: str) -> str:
    """
    Normalize the answer by lowercasing, removing punctuation, articles,
    and extra whitespace.

    Based on:
    https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))
    
def verify_one_question(response: str, target: str, use_exact_match: bool = False) -> float:
    """
    Verify a single question response against ground truth.
    
    Args:
        response: The response text for one question
        ground_truth: Ground truth data for this question
        use_exact_match: If True, use exact match; if False, use contains match
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not response or not target:
        return 0.0
    
    # check if the response is a list of answers
    parsed_labels: Union[List, str]
    try:
        parsed = json.loads(target)
        parsed_labels = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        # Fallback: treat label as raw string or list-of-strings
        if isinstance(target, list):
            parsed_labels = target
        else:
            parsed_labels = [str(target).strip()]
    
    for label in parsed_labels:
        # Normalize both strings for comparison
        response_normalized = normalize_answer(extract_boxed_answer_from_response(response.strip()))
        target_normalized = normalize_answer(str(label).strip())
        
        if use_exact_match:
            # Exact match after normalization
            if response_normalized == target_normalized:
                return 1.0
        else:
            # Contains match - check if ground truth is contained in response
            if target_normalized.lower() in response_normalized.lower():
                return 1.0
    return 0.0



def compute_multi_question_reward(
    response: str, 
    ground_truth: str, 
    query: Optional[str] = None,
    use_exact_match: bool = False,
    reward_type: str = "finegrained",
    ) -> Dict[str, Any]:
    """
    Compute finegrained multi-question reward with simple verifiable scoring and spans.
    
    Args:
        response: The full response containing multiple answers
        ground_truth: Dictionary containing ground truth data
        use_exact_match: If True, use exact match; if False, use contains match
        
    Returns:
        Dict with:
            - finegrained_scores: List of (score, (start_char, end_char), reward_group_id, response_idx) tuples
            - log_values: Dict of metrics for logging
    """
    # Get verifiable reward scores for each question
    ground_truth_per_question = extract_ground_truth_per_question(ground_truth)
    num_questions = len(ground_truth_per_question)
    
    # Split the response into individual question components for span generation
    sub_responses, spans, num_valid_spans = split_response_and_get_spans(response, num_questions)
    
    # Compute finegrained scores for each question
    finegrained_scores = []
    for i, (sub_response, gt, span) in enumerate(zip(sub_responses, ground_truth_per_question, spans)):
        sub_score = verify_one_question(sub_response, gt, use_exact_match)

        group_id = 0  # rulin: assume the same group for all questions
        finegrained_scores.append(
            FinegrainedScore(
                score=sub_score,
                effective_spans=span,
                reward_group_id=group_id,  
                reward_group_name=f"question_{group_id}",
            )
        )
    
    averaged_score = sum(item.score for item in finegrained_scores) / len(finegrained_scores)
    
    # Create log values for tracking
    log_values = {
        **{f"question_{i}_accuracy": item.score for i, item in enumerate(finegrained_scores)},
        "num_questions": num_questions,
        "averaged_accuracy": averaged_score,
        "num_valid_spans": num_valid_spans,
    }
    
    if reward_type == "finegrained":
        print(f"🎀 finegrained_scores: {log_values}")
        return {
            "finegrained_scores": finegrained_scores,
            "log_values": log_values,
        }
    elif reward_type == "averaged":
        print(f"🎀 averaged_score: {averaged_score}")
        return {
            "score": averaged_score,
            "log_values": log_values,
        }
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")


if __name__ == "__main__":
    response = """<think>I will start by researching question 1 about when Metro Pictures adopted the Goldwyn mascot and motto.</think>
<search>when metro pictures took over two other companies to form mgm</search>

<snippets id=c3873eb5>
Metro-Goldwyn-Mayer United Artists in 1981. MGM ramped up internal production, as well as keeping production going at UA, which included the lucrative James Bond film franchise. It also incurred significant amounts of debt to increase production. The studio took on additional debt as a series of owners took charge in the 1980s and early 1990s. In 1986, Ted Turner bought MGM, but a few months later, sold the company back to Kerkorian to recoup massive debt, while keeping the library assets for himself. The series of deals left MGM even more heavily in debt. MGM was bought by Pathé Communications (led

United Artists acquired. Metro-Goldwyn-Mayer acquired the studio in 1981 for a reported $350 million. On September 22, 2014, MGM acquired a controlling interest in Mark Burnett and Roma Downey's entertainment companies One Three Media and Lightworkers Media, then merged them to revive United Artists' TV production unit as United Artists Media Group (UAMG). However, on December 14 of the following year, MGM wholly acquired UAMG and folded it into MGM Television. UA was revived yet again in 2018. Pickford, Chaplin, Fairbanks, and Griffith incorporated UA as a joint venture on February 5, 1919. Each held a 25 percent stake in the preferred

The Samuel Goldwyn Company operated as MGM's specialty films unit. A month later, Samuel Goldwyn Jr. sued MGM and Metromedia, claiming that he was abruptly let go of the company despite promises that he would continue to run it under different ownership. Another concern in the lawsuit was the use of the Goldwyn name, with the defendants being accused of “palming off specialized films produced or acquired by” the unit as though the plaintiff was still involved in its management. Goldwyn Films changed its name to G2 Films in January 1999 as part of the settlement. In July 1999, G2 Films was renamed United
</snippets> 

<answer1>\boxed{1981}</answer1>

Now, I will move on to the second question about St. David’s Day.

<think>I will find out when St. David’s Day falls every year.</think>
<search>st. david's day date</search>

<snippets id=d1349364>
Saint David's Day Saint David's Day Saint David's Day (, ) is the feast day of Saint David, the patron saint of Wales, and falls on 1 March, the date of Saint David's death in 589 AD. The feast has been regularly celebrated since the canonisation of David in the 12th century (by Pope Callistus II), though it is not a national holiday in the UK. Traditional festivities include wearing daffodils and leeks, recognised symbols of Wales and Saint David respectively, eating traditional Welsh food including cawl and Welsh rarebit, and women wearing traditional Welsh dress. An increasing number of cities and towns

Saint David's Day important centre in Wales. The date of Saint David's death is believed to be 1 March 589. His final words to the community of monks were: "Brothers be ye constant. The yoke which with single mind ye have taken, bear ye to the end; and whatsoever ye have seen with me and heard, keep and fulfil." For centuries, 1 March has been a national festival. Saint David was recognised as a national patron saint in the 12th century at a peak time of Welsh resistance to the Normans. He was canonised by Pope Callixtus II in 1120. The 17th-century diarist

Bank holiday this as an official bank holiday there. After the election of the Coalition Government in May 2010, the Department of Culture, Media and Sport launched a pre-consultation in 2011 which included the suggestion of moving the May Day Bank Holiday to October, to be a "UK Day" or "Trafalgar Day" (21 October) or to St David's Day and St George's Day. It is suggested that a move from the May bank holiday to a St Piran's Day bank holiday in Cornwall, on 5 March, would benefit the Cornish economy by £20–35 million. During the sterling crisis of 1968, Prime Minister
</snippets> 

<answer2>\boxed{March 1}</answer2>

For the third question, I'll have to look up the high priest of Judea during the time of Christ's arrest and execution.

<think>I need to find out the name of the high priest at the time of Jesus' arrest and execution.</think>
<search>high priest during jesus' arrest and execution</search>

<snippets id=96db4d55>
Caiaphas Caiaphas Joseph Caiaphas, known simply as Caiaphas (; ) in the New Testament, was the Jewish high priest who organized the plot to kill Jesus. Caiaphas was involved in the Sanhedrin trial of Jesus. The primary sources for Caiaphas' life are the New Testament and the writings of Josephus. Outside of his interactions with Jesus, little else is known about his tenure as high priest. The 1st-century Jewish historian Josephus is considered the most reliable extra-biblical literary source for Caiaphas. His works contain information on the dates for Caiaphas' tenure of the high priesthood, along with reports on other high

Good Friday carry out a sentence of death (). Pilate questioned Jesus and told the assembly that there was no basis for sentencing. Upon learning that Jesus was from Galilee, Pilate referred the case to the ruler of Galilee, King Herod, who was in Jerusalem for the Passover Feast. Herod questioned Jesus but received no answer; Herod sent Jesus back to Pilate. Pilate told the assembly that neither he nor Herod found Jesus to be guilty; Pilate resolved to have Jesus whipped and released (). Under the guidance of the chief priests, the crowd asked for Barabbas, who had been imprisoned for

Crucifixion of Jesus with myrrh or gall to drink after saying "I am thirsty". He was then hung between two convicted thieves and, according to the Gospel of Mark, died some six hours later. During this time, the soldiers affixed a sign to the top of the cross stating "Jesus of Nazareth, King of the Jews" which, according to the Gospel of John, was written in three languages. They then divided his garments among themselves and cast lots for his seamless robe, according to the Gospel of John. According to the Gospel of John after Jesus' death, one soldier pierced his side with
</snippets> 

<answer3>\boxed{Caiaphas}</answer3>

For the fourth question, I need to find out which US state is called the 'Old Dominion'.

<think>I will search for information on the state that is called the 'Old Dominion'.</think>
<search>the old dominion us state</search>

<snippets id=66d7cc61>
Culture of the Southern United States "Redeemer" government in 1876. Many legacies of its Virginia heritage remain, such as county and local place names. The state constitution is based on the antebellum constitution of Virginia. As recently as 2007 an 1849 Virginia statute was used in a county prosecution. Historic plantation houses are found throughout the state, legacies of its antebellum origins. West Virginia was the last slave state admitted to the Union. The state legislature consists of a senate and a house of delegates. The state government belongs to the Southern Governors Association and the Southern Legislative Conference. It is the 7th most Protestant state

Morris v. United States the District's border with Virginia (just as Maryland's southern border remained in doubt). Shortly after the creation of the District of Columbia, the United States government sold certain plots of land to James M. Marshall; his brother, John Marshall (later Chief Justice of the United States); John L. Kidwell; the Chesapeake and Ohio Canal Company; and several others. Maryland and Virginia agreed to arbitrate their dispute, and in 1877 the Black-Jenkins Award (as the decision of the arbitration panel is known) placed Virginia's boundary with Maryland at the low-water mark on the Virginia side of the Potomac River. In 1882,

West Virginia some Unionist counties of northwestern Virginia decided to break away from Virginia, although they included many secessionist counties in the new state. West Virginia was admitted to the Union on June 20, 1863, and was a key border state during the war. West Virginia was the only state to form by separating from a Confederate state, the first to separate from any state since Maine separated from Massachusetts, and was one of two states admitted to the Union during the American Civil War (the other being Nevada). While a portion of its residents held slaves, most of the residents were
</snippets> 

<answer4>\boxed{Virginia}</answer4>

For the final question, I'll be finding out which Australian state Darwin is the capital of.

<think>I need to research which Australian state Darwin is the capital of.</think>
<search>which state is darwin the capital of</search>

<snippets id=c81dff6e>
Darwin, Northern Territory Darwin, Northern Territory Darwin ( ) is the capital city of the Northern Territory of Australia, situated on the Timor Sea. It is the largest city in the sparsely populated Northern Territory, with a population of 145,916. It is the smallest and most northerly of the Australian capital cities, and acts as the Top End's regional centre. Darwin's proximity to South East Asia makes it a link between Australia and countries such as Indonesia and East Timor. The Stuart Highway begins in Darwin, extends southerly across central Australia through Tennant Creek and Alice Springs, concluding in Port Augusta, South Australia.

Darwin, Northern Territory Australia, New Zealand, Singapore, Thailand, United Arab Emirates, and the United States. Darwin, Northern Territory Darwin ( ) is the capital city of the Northern Territory of Australia, situated on the Timor Sea. It is the largest city in the sparsely populated Northern Territory, with a population of 145,916. It is the smallest and most northerly of the Australian capital cities, and acts as the Top End's regional centre. Darwin's proximity to South East Asia makes it a link between Australia and countries such as Indonesia and East Timor. The Stuart Highway begins in Darwin, extends southerly across central Australia

Darwin, Northern Territory itself—forming one corner, the newer northern suburbs another, and the eastern suburbs, progressing towards Palmerston, forming the third. The older part of Darwin is separated from the newer northern suburbs by Darwin International Airport and RAAF Base Darwin. Palmerston is a satellite city east of Darwin that was established in the 1980s and is one of the fastest growing municipalities in Australia. The rural areas of Darwin including Howard Springs, Humpty Doo and Berry Springs are experiencing strong growth. Darwin's central business district is bounded by Daly Street in the north-west, McMinn Street in the north-east, Mitchell Street on the
</snippets> 

Darwin, Northern Territory Darwin is located in Australia, making it the capital of the Northern Territories. It is situated on the Timor Sea and is considered the most northerly of the Australian capital cities.

<answer5>\boxed{Northern Territory}</answer5>"""
    ground_truth = """["Art for art\'s sake", "Ars gratia artis", "Ars Gratia Artis", "Art For Art\'s Sake", "L\'art pour l\'art", "Art for Art\'s Sake"]; ["Mar 01", "1st of March", "Historical anniversaries/March 1", "1 March", "March 1st", "March 01", "Mar 1", "1st March", "March 1"]; ["Joseph ben Caiaphas", "Kajafas", "Caiphas", "Joseph Caiaphas", "Caiaphas", "Kaiphas", "Caifa", "Joseph Caiphas", "Kajaphas"]; ["The Virginia", "VIRGINIA"]; ["Northern Territory", "Northern territory", "NORTHERN TERRITORY", "North Territory", "The Northern Territory", "NorthernTerritory", "Northern Territory of Australia", "AU-NT", "Northern Territory, Australia", "South Australia (Northern Territory)", "Local government in the Northern Territory", "Northern Territorian", "Northern Territories, Australia", "North territory"]"""
    print(compute_multi_question_reward(response, ground_truth, query=None, reward_type="averaged"))