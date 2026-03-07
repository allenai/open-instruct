import json
import re

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

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

# Define the templates for different judge types
general_quality_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""


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

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""


general_quality_ref_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.

Notes:
- Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Reference Gold Answer]
{label}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""

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
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}
"""

web_instruct_general_verifier_template = """User: ### Question: {input}


### Ground Truth Answer: {label}


### Student Answer: {output}


For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output Final Decision: No. Assistant:"""

compass_verifier_template = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

# TODO: just a copy (need to be updated)
creative_writing_template = """
### Task Description
You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
Question: {{input}}
Provided Answer: {{output}}
Reference Answer:{{label}}
Correct:
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}
"""

# TODO: just a copy (need to be updated)
refusal_template = """
### Task Description
You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
Question: {{input}}
Provided Answer: {{output}}
Reference Answer:{{label}}
Correct:
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}
"""


def build_messages(user_prompt: str, system_prompt: str | None = None):
    """
    Build the message payload for the model evaluation.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def extract_score_from_string(score_str: str) -> float:
    """Extract numerical score from string response."""
    # Handle rating formats like "4/5"
    ratio_matches = re.findall(r"(\d+)\/(\d+)", score_str)
    if ratio_matches:
        numerator, denominator = ratio_matches[0]
        return float(numerator) / float(denominator)

    # Try to handle percentage expressions
    percent_matches = re.findall(r"(\d+\.?\d*|\.\d+)%", score_str)
    if percent_matches:
        return float(percent_matches[0]) / 100.0

    # Try to find numerical values in the string
    matches = re.findall(r"(\d+\.?\d*|\.\d+)", score_str)
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


def extract_score_web_instruct(score_str: str) -> "tuple[str, float]":
    """Extractor based on web instruct format"""
    if "final decision: yes" in score_str.lower():
        return score_str, 1.0
    elif "final decision: no" in score_str.lower():
        return score_str, 0.0
    logger.warning(f"Could not parse score from: {score_str}, defaulting to 0.0")
    return score_str, 0.0


def extract_score_compass_verifier(score_str: str) -> "tuple[str, float]":
    """Extractor for CompassVerifier's A/B/C judgments."""
    cleaned = score_str.strip()

    boxed_matches = re.findall(r"boxed\{([A-C])\}", cleaned, flags=re.IGNORECASE)
    if boxed_matches:
        return score_str, 1.0 if boxed_matches[-1].upper() == "A" else 0.0

    if cleaned.upper() in {"A", "B", "C"}:
        return score_str, 1.0 if cleaned.upper() == "A" else 0.0

    final_judgment = cleaned.split("Final Judgment:")[-1]
    letter_matches = re.findall(r"\b([A-C])\b", final_judgment, flags=re.IGNORECASE)
    if letter_matches:
        return score_str, 1.0 if letter_matches[-1].upper() == "A" else 0.0

    lowered = cleaned.lower()
    if "incorrect" in lowered or "invalid" in lowered or "incomplete" in lowered or "refusal" in lowered:
        return score_str, 0.0
    if "correct" in lowered:
        return score_str, 1.0

    logger.warning(f"Could not parse CompassVerifier score from: {score_str}, defaulting to 0.0")
    return score_str, 0.0


def extract_json_score_with_fallback(score_str: str) -> "tuple[str, float]":
    """Extractor based on json score with fallback"""
    try:
        # Strip markdown code blocks if present
        cleaned_str = score_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:]  # Remove ```json
        elif cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:]  # Remove ```

        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]  # Remove trailing ```

        # escape newlines
        cleaned_str = cleaned_str.replace("\r\n", "\n").replace("\n", "\\n")
        # escape backslashes
        cleaned_str = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned_str)

        cleaned_str = cleaned_str.strip()

        try:
            data = json.loads(cleaned_str)
            reasoning = data.get("REASONING", "")
            score = float(data.get("SCORE", 0.0))
        except json.JSONDecodeError as e:
            score_match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned_str)
            if score_match:
                score = float(score_match.group(1))
                reasoning = cleaned_str
            else:
                raise ValueError() from e
        return reasoning, score
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(f"Could not parse score from due to invalid json: {score_str}, defaulting to 0.0")
        return score_str, 0.0


def extract_score_with_fallback_max_10(score_str: str) -> "tuple[str, float]":
    """Extractor based on score with fallback"""
    reasoning, score = extract_json_score_with_fallback(score_str)
    return reasoning, score / 10.0


JUDGE_PROMPT_MAP = {
    "quality": general_quality_template,
    "quality_rubric": general_quality_rubric_template,
    "quality_ref": general_quality_ref_template,
    "safety": safety_template,
    "factuality": factuality_template,
    "creative_writing": creative_writing_template,
    "refusal": refusal_template,
    "web_instruct_general_verifier": web_instruct_general_verifier_template,
    "compass_verifier": compass_verifier_template,
}

EXTRACTOR_MAP = {
    "quality": extract_score_with_fallback_max_10,
    "quality_rubric": extract_json_score_with_fallback,
    "quality_ref": extract_score_with_fallback_max_10,
    "safety": extract_json_score_with_fallback,
    "factuality": extract_json_score_with_fallback,
    "creative_writing": extract_json_score_with_fallback,
    "refusal": extract_json_score_with_fallback,
    "web_instruct_general_verifier": extract_score_web_instruct,
    "compass_verifier": extract_score_compass_verifier,
}

REQUIRED_OUTPUT_SUFFIX_MAP = {
    "quality": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "quality_rubric": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "quality_ref": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "safety": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "factuality": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "creative_writing": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "refusal": 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}',
    "web_instruct_general_verifier": None,
    "compass_verifier": None,
}
