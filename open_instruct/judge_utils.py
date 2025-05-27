from textwrap import dedent
import json
import re
import logging

logger = logging.getLogger(__name__)

PRICE_PER_TOKEN = {
    "gpt-4": {"input": 0.00003, "output": 0.00006}, 
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002}, 
    "gpt-4-1106-preview": {"input": 0.00001, "output": 0.00003},
    "gpt-4o": {"input": 0.0000025, "output": 0.000001},
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o-standard": {"input": 0.0000025, "output": 0.000001},
    "gpt-4.1": {"input": 0.000002, "output": 0.000008},
    "gpt-4.1-standard": {"input": 0.000002, "output": 0.000008},
    "gpt-4.1-mini-standard": {"input": 0.0000004, "output": 0.0000016},
    "o3": {"input": 0.00001, "output": 0.00004},
    "o3-standard": {"input": 0.00001, "output": 0.00004},
    "claude-sonnet": {"input": 0.000003, "output": 0.000015}, 
    "deepseek-chat": {"input": 0.00000007, "output": 0.000001}, 
    "deepseek-reasoner": {"input": 0.00000014, "output": 0.000002}, 
    "claude-3-7-sonnet-20250219": {"input": 0.000003, "output": 0.000015},
}

def extract_score_from_string(score_str: str) -> float:
    """Extract numerical score from string response.""" 
    # Handle rating formats like "4/5"
    ratio_matches = re.findall(r'(\d+)\/(\d+)', score_str)
    if ratio_matches:
        numerator, denominator = ratio_matches[0]
        return float(numerator) / float(denominator)
       
    # Try to handle percentage expressions
    percent_matches = re.findall(r'(\d+\.?\d*|\.\d+)%', score_str)
    if percent_matches:
        return float(percent_matches[0]) / 100.0

    
    # Try to find numerical values in the string
    matches = re.findall(r'(\d+\.?\d*|\.\d+)', score_str)
    if matches:
        return float(matches[0])
    
    # If parsing fails, check for binary indicators
    if any(word in score_str.lower() for word in ["yes", "correct", "good", "true", "pass"]):
        return 1.0
    elif any(word in score_str.lower() for word in ["no", "incorrect", "bad", "false", "fail"]):
        return 0.0
    else:
        logger.warning(f"Could not parse score from: {score_str}, defaulting to 0.0")
        return 0.0

# Define the templates for different judge types
general_quality_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below. 

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Begin your evaluation by providing a short explanation. 
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10.

[Query]
{input}

[Response]
{output}

[Your judgement]"""


general_quality_rubric_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below. You are shown the user query, assistant response, and an evaluation criteria.

Notes:
- Each response must be evaluated against these criteria independently.
- An overall score of 1 requires meeting all criteria.
- An overall score of 0 in any single category results in the output being classified as bad (0).
- Begin your evaluation by providing a short explanation. 
- Be as objective as possible. After providing your explanation, please output the overall score of 0 or 1.


### Evaluation Criteria
Use the following criteria to evaluate the response:
{label}

[Query]
{input}

[Response]
{output}

[Your judgement]"""


general_quality_ref_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the user query displayed below. Judge whether the provided answer is good by comparing it to the reference answer. 

Notes:
- Besides comparing to the referennce answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and appropriate level of detail of the response.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- Begin your evaluation by providing a short explanation. 
- Be as objective as possible. After providing your explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Query]
{input}

[Answer]
{output}

[Reference Answer]
{label}

[Your judgement]"""

# create empty template (TODO, incomplete)
safety_template = """
### Task Description \
"""

factuality_template = """
### Task Description
You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".

NOTES:
1- Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".

Question: {{input}}
Provided Answer: {{output}}
Reference Answer:{{label}}
Correct:
"""

### TODO: just a copy (need to be updated)
creative_writing_template = """
### Task Description
You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
Question: {{input}}
Provided Answer: {{output}}
Reference Answer:{{label}}
Correct:
"""

### TODO: just a copy (need to be updated)
refusal_template = """
### Task Description
You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
Question: {{input}}
Provided Answer: {{output}}
Reference Answer:{{label}}
Correct:
"""


JUDGE_PROMPT_MAP = {
    "quality": general_quality_template,
    "quality_rubric": general_quality_rubric_template,
    "quality_ref": general_quality_ref_template,
    "safety": safety_template,
    "factuality": factuality_template,
    "creative_writing": creative_writing_template,
    "refusal": refusal_template,
}
