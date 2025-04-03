from textwrap import dedent
import json
from judges.base import BaseJudge, Judgment




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
AI assistant to the user query displayed below. You are shown the user query, assistant response, and the evaluation criteria.

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

# rate 0 to 10
# general_quality_rubric_template = """
# ### Task Description
# Please act as an impartial judge and evaluate the quality of the response provided by an
# AI assistant to the user query displayed below. You are shown the user query, assistant response, and the evaluation criteria.

# Notes:
# - Each response must be evaluated against these criteria independently.
# - An overall score of 1 requires meeting all criteria.
# - An overall score of 0 in any single category results in the output being classified as bad (0).
# - Begin your evaluation by providing a short explanation. 
# - Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10.


# ### Evaluation Criteria
# Use the following criteria to evaluate the response:
# {label}

# [Query]
# {input}

# [Response]
# {output}

# [Your judgement]"""

general_quality_ref_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the user query displayed below. Judge whether the provided answer is good by comparing it to the reference answer. 

Notes:
1- Besides comparing to the referennce answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable.
3- Begin your evaluation by providing a short explanation. 
4- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10.

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
