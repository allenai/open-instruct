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

procedure_judge_template = """You are given a goal and two lists of steps, L1 and L2. L1 is one correct procedure that is guaranteed to achieve the goal. L2 is a candidate procedure whose correctness needs to be determined. Your task is to determine whether L2 has any **critical failures**, using the goal and L1 as the reference.

# Important guidelines

- **L1 as reference**
  L1 is guaranteed to succeed but may not be the only correct method. Use it as a reliable reference, not the exclusive solution.

- **Definition of critical failure**
  A critical failure is where:
  - An L2 step directly contradicts or significantly diverges from an L1 step, preventing the goal from being achieved.
  - An L2 step is inconsistent with another step within L2.
  - An L2 step repeats one or more previous steps in L2, where no such repetition exists in L1.
  - An L2 step introduces an action not present in L1 that is unnecessary or counterproductive.
  - An essential L1 step required to achieve the goal is completely omitted in L2, with no equivalent or implied action present.

- **Acceptable variations**
  Cases like the following do not count as failures:
  - Differences between L1 and L2 in style or wording.
  - Extra steps that are neutral or practical (e.g., cleanup, storage).
  - Reasonable implicit equivalence (e.g., "follow manufacturer's instructions" instead of explicitly identifying the wheel finish).

- **Style**
  Ignore differences in style, wording, or level of detail. Focus only on the actions presented in L2. Extra verbosity or added detail does not make L2 better or worse.

- **External knowledge**
  Base all decisions only on the provided Goal and L1. Minimize reliance on outside knowledge as much as possible.

# Examples

Below are examples of what qualifies as a critical failure, as well as examples of what does not. To keep things concise, the L1 and L2 cases are shown in summarized form. Please read through them carefully to understand how to make the distinction. Keep in mind that these examples are not an exhaustive list of all possible failures for each L2.

## Examples of critical failures

Note: The following examples are not listing all failures present in each L2; it's only for demonstration purposes.

Goal: Prepare Indian-style red lentil dhal for 8 portions using an oven and skillet.
Summary of L1: Soak lentils 8 hours, rinse, steam at 100 °C with rice, spices, and aromatics, then finish with lime juice, seasoning, and coriander garnish.
Summary of L2: Soak lentils only 30 minutes, then fry onions, garlic, chili, cumin, and salt in ghee, add lentils with water, and simmer until soft.
Example failure: L2 soaks lentils for only 30 minutes, whereas L1 soaks for 8 hours. This is a critical difference in time.
Example failure: L2 omits the oven entirely, using only stovetop simmering, which deviates from L1's oven-based preparation method and contradicts the goal.

Goal: To construct a traditional wooden Jacob's Ladder toy using wood, ribbon, and small nails.
Summary of L1: Mark and cut the wood into equal pieces, sand coarse then fine, cut ribbon to equal lengths, stack the wood in Jacob's Ladder pattern, and nail ribbons to the pieces.
Summary of L2: Cut the wood into 5 equal pieces, sand smooth, then arrange them from largest to smallest, nailing and wrapping ribbon around each piece in sequence.
Example failure: L2 contradicts itself; if the 5 pieces are of equal size, there is no largest or smallest piece.

Goal: To treat head lice by applying a tea tree oil and apple cider vinegar solution to the hair.
Summary of L1: Mix tea tree oil with apple cider vinegar, wash hair, apply solution, cover 15 minutes, rinse, then comb with a fine-tooth comb.
Summary of L2: Wash hair with shampoo, apply diluted tea tree oil–vinegar spray under a cap for 1 hour, comb, and repeat treatment over 2 weeks, wash hair with shampoo.
Example failure: L2 step 7 repeats the shampooing step almost verbatim, a redundancy not present in L1.

Goal: Prepare an alkyl chloride from a primary or secondary alcohol using thionyl chloride to avoid acid and rearrangements.
Summary of L1: Place alcohol in a flask, add thionyl chloride, reflux, cool, then separate and dry the alkyl chloride with a drying agent.
Summary of L2: Add alcohol and thionyl chloride to a flask, then add the drying agent, attach condenser, reflux, cool, and filter off the drying agent.
Example failure: L2 adds the drying agent to the flask before heating the flask, while L1 uses the drying agent at the very end.

Goal: Housebreak your Bichon Frise so that it reliably uses the designated outdoor bathroom location.
Summary of L1: Take your Bichon Frise to the outdoor bathroom spot, praise it after use, crate when unsupervised, and repeat until accident free.
Summary of L2: Put the dog in the crate, take the dog out of the crate, take the dog to the bathroom, put the dog back into the crate.
Example failure: L2 omits praising the dog after outdoor bathroom use, removing the positive reinforcement step that is critical in L1 for reliable housebreaking.

## Examples of acceptable variations that do not count as failures

Goal: Prepare Ambrosia Fruit Dip using cream cheese, yogurt, vanilla extract, grated lemon rind, and Equal sweetener.
Summary of L1: Blend cream cheese and yogurt until smooth, add vanilla, lemon rind, and sweetener, mix well, and chill in refrigerator.
Summary of L2: Combine cream cheese, yogurt, vanilla, and sweetener, beat until smooth, add lemon rind, chill, then serve with fruit, enjoy, clean up, and store leftovers.
Acceptable variation: L2 last step (storing leftovers) is not in L1, but it is an extra practical step that is reasonable and does not hurt the process.

Goal: Prepare a package for shipping so that its contents arrive in good condition.
Summary of L1: Choose a strong box, wrap and cushion items, fill empty space, close and tape box, attach label, and remove old labels.
Summary of L2: Place items in box with cushioning, tape securely, attach and verify label, seal seams, mark fragile if needed, and send to shipping service.
Acceptable variation: L2 omits removing old labels, but this is not critical since it is reasonable to assume a new box without old labels.

Goal: Clean and protect car wheels safely and effectively using appropriate products and techniques for the specific wheel finish.
Summary of L1: Identify wheel finish, choose a safe cleaner, spray from bottom up, agitate with mitt/brush, and rinse thoroughly.
Summary of L2: Follow manufacturer's cleaning recommendations, wash with mitt and cleaner, rinse, polish with metal polish, and apply protectant.
Acceptable variation: L1 explicitly requires identifying the wheel finish, while L2 implies this through reading the manufacturer's recommendations — a reasonable equivalent rather than a critical omission.

Goal: Create distressed terra cotta pots as baby shower favors, each with an herb seed packet in a stamped muslin bag.  
Summary of L1: Paint pots with a base coat, dry, add a second coat, dry overnight, sand for a distressed look, add pebbles, tie twine with a thank-you note, stamp “GROW” on muslin bags, insert herb seed packets, and place the bags next to each pot.  
Summary of L2: Paint pots in a contrasting color and dry, lightly sand, add pebbles and soil, tie twine with a handwritten thank-you card, stamp and label muslin bags with the herb name, fill with seed packets, tie shut, and place the bag in each pot.  
Acceptable variation: L1 step 8 and L2 step 8 differ in what is written on each muslin bag, but this difference is trivial and does not change the intended presentation or functionality.

Goal: Create a glowing mixture by combining Mountain Dew, dishwashing liquid, hydrogen peroxide, baking soda, and the contents of a glowstick in a glass beaker.  
Summary of L1: Pour Mountain Dew into a beaker, add dishwashing liquid, hydrogen peroxide, and baking soda, cut open a glowstick and add its contents.  
Summary of L2: Cut open a glowstick and add its contents to a beaker, add dishwashing liquid and baking soda, stir, add hydrogen peroxide, stir again, then add Mountain Dew and stir once more.  
Acceptable variation: L1 adds Mountain Dew at the beginning while L2 adds it at the very end, but this mismatch is not critical and does not affect the outcome.

# Input data

Goal:
{goal}

L1:
{reference_steps}

L2:
{steps}

# Output format

- Identify **all** critical failures in the given L2, and return them as a list called "critical_failures".
- The "failure" field should provide a concise and clear explanation of what the failure is.
- Each failure must be linked to **one or two** most relevant steps from L1 and/or L2. Record these in the "L1_steps" and "L2_steps" fields as lists of step numbers. Only in rare, exceptional cases—with clear justification—should you associate more than two steps.
- If no failures are found, return "critical_failures": [].

Return your response in the following JSON format:

{{
  "critical_failures": [
    {{
      "failure": "<string>",
      "L1_steps": [<int>],
      "L2_steps": [<int>]
    }},
    ...
  ]
}}"""


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


def extract_binary_from_critical_failures(score_str: str) -> "tuple[str, float]":
    """
    Parse model JSON with a top-level "critical_failures" array.
    Return 1.0 if it is an empty list; otherwise 0.0.
    """
    try:
        cleaned_str = score_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:]
        elif cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:]
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]
        cleaned_str = cleaned_str.strip()

        data = json.loads(cleaned_str)
        failures = data.get("critical_failures", []) if isinstance(data, dict) else []
        score = 1.0 if isinstance(failures, list) and len(failures) == 0 else 0.0
        return cleaned_str, score
    except Exception:
        logger.warning("Failed to parse critical_failures for binary extraction; defaulting to 0.0")
        return score_str, 0.0


JUDGE_PROMPT_MAP = {
    "quality": general_quality_template,
    "quality_rubric": general_quality_rubric_template,
    "quality_ref": general_quality_ref_template,
    "safety": safety_template,
    "factuality": factuality_template,
    "creative_writing": creative_writing_template,
    "refusal": refusal_template,
    "web_instruct_general_verifier": web_instruct_general_verifier_template,
    "procedure_judge": procedure_judge_template,
    # Aliases reuse the same template
    "procedure_judge_binary": procedure_judge_template,
    "procedure_judge_ratio": procedure_judge_template,
    "procedure_judge_ratio_2": procedure_judge_template,
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
    # Procedure judges: binary uses dedicated extractor; ratio and plain use JSON fallback (score computed elsewhere)
    "procedure_judge": extract_json_score_with_fallback,
    "procedure_judge_binary": extract_binary_from_critical_failures,
    "procedure_judge_ratio": extract_json_score_with_fallback,
    "procedure_judge_ratio_2": extract_json_score_with_fallback,
}
