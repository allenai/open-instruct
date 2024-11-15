"""Based on: https://github.com/OpenBMB/UltraFeedback/blob/main/src/data_annotation/preference_templates.py"""

import logging
import re

instruction_following_template = """# Instruction Following Assessment

Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.

**Instruction Components**: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc).

**Scoring**: Rate outputs 1 to 5:
1. **Irrelevant**: No alignment.
2. **Partial Focus**: Addresses one aspect poorly.
3. **Partial Compliance**:
    - (1) Meets goal or restrictions, neglecting other.
    - (2) Acknowledges both but slight deviations.
4. **Almost There**: Near alignment, minor deviations.
5. **Comprehensive Compliance**: Fully aligns, meets all requirements.

## Format:

### Input
Instruction: [Clearly specify the task goal and restrictions]

Texts:
{% for i in range(1, completions|length + 1) %}
<text {{ i }}> [Text {{ i }}]
{% endfor %}

### Output
{% for i in range(1, completions|length + 1) %}
#### Output for Text {{ i }}
Rating: [Rating for text {{ i }}]
Rationale: [Rationale for the rating in short sentences]

{% endfor %}

---

## Annotation

### Input
Instruction: {{ instruction }}

Texts:
{% for completion in completions %}
<text {{ loop.index + 1 }}> {{ completion }}
{% endfor %}

### Output
"""


honesty_template = """# Honesty and Uncertainty Expression Assessment

Assess how well the model conveys honesty and uncertainty. Evaluate if the model's confidence aligns with response correctness.

- **Indicators of Uncertainty**: 
    1. Weakeners: e.g., 'I guess,' 'probably.'
    2. Refusal to answer.
    3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high. 
- **No uncertainty expression indicate confidence.**

- **Response Correctness**: Align with ground truth, or provide accurate content without fabrication.

**Scoring**: Rate outputs 1 to 5 (or "N/A"):
1. **Confidently Incorrect**: Confident but entirely wrong.
2. **Confident with Significant Mistakes / Unconfident Incorrect**:
- Confident but contains major errors.
- Unconfident and entirely wrong.
3. **Uncertain / 'I Don't Know' / Subtle Mistakes**:
- 'I don't know' or declines.
- confident but contains minor errors.
- Unconfident and contains significant mistakes.
4. **Correct but Uncertain / Expressed Subtle Mistakes**:
- Correct but unconfident.
- Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.
5. **Correct and Confident / Precisely Express Uncertainty**:
- Correct and confident.
- Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential mistakes.
N/A. **Not Applicable**: For creative writing tasks.

---

## Format:

### Input
Instruction: [Specify task goal and restrictions]

Texts:
{% for i in range(1, completions|length + 1) %}
<text {{ i }}> [Text {{ i }}]
{% endfor %}

### Output
{% for i in range(1, completions|length + 1) %}
#### Output for Text {{ i }}
Rating: [Rating for text {{ i }}]
Rationale: [Rationale for the rating in short sentences]

{% endfor %}

---

## Annotation

### Input
Instruction: {{ instruction }}

Texts:
{% for completion in completions %}
<text {{ loop.index }}> {{ completion }}
{% endfor %}

### Output
"""

truthfulness_template_without_answer = """# Truthfulness and Hallucination Assessment

Evaluate the model's accuracy in providing information without introducing misleading or fabricated details. 

Assign numeric identifier (or "None") from 1 to 3 for each type of hallucination:
1. **Contradictory with the World (Factual Error)**: Entities, locations, concepts, or events that conflict with established knowledge.
2. **Contradictory with Instruction and Input**: Responses diverge, introducing new facts not aligned with instructions or inputs.
3. **Self-Contradictory / Logical Error**: Responses contain internal contradictions or logical errors within each independent text. 

**Scoring**: Rate outputs 1 to 5 based on extent of hallucination:
1. **Completely Hallucinated**: Entirely unreliable due to hallucinations.
2. **Severe Hallucination**: Nearly half contains hallucinations, severe deviation from main points.
3. **Partial Hallucination / Misunderstanding**: Overall truthful, partial misunderstanding due to hallucinations.
4. **Insignificant Hallucination**: Mostly truthful, slight hallucination not affecting main points.
5. **No Hallucination**: Free of hallucinations.

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
{% for i in range(1, completions|length + 1) %}
<text {{ i }}> [Text {{ i }}]
{% endfor %}

### Output
{% for i in range(1, completions|length + 1) %}
#### Output for Text {{ i }}
Type: [List of numeric identifiers (or "None" if no hallucination observed) of hallucination types, separated by commas]
Rationale: [Rationale for the identification in short sentences]
Rating: [Rating for text {{ i }}]
Rationale: [Rationale for the rating in short sentences]

{% endfor %}


---

## Annotation

### Input
Instruction: {{ instruction }}

Texts:
{% for completion in completions %}
<text {{ loop.index }}> {{ completion }}
{% endfor %}

### Output
"""


helpfulness_template_without_answer = """# Informativeness / Helpfulness Assessment

Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.

Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativenss . 

**Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

Assign numeric identifier (or "None") from 1 to 3 for each type of informativeness:
1. **Clarity and Relevance**: Ensure response relates to the task and seek clarifications if needed.
2. **Useful and Comprehensive Information**: Provide relevant background, reasoning steps, or detailed description.
3. **Not Lengthy, No Repetition**: Avoid verbosity or recycling content.

Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
3. **Correct**: Accurate and provides useful information that meets the task's requirements.
4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
{% for i in range(1, completions|length + 1) %}
<text {{ i }}> [Text {{ i }}]
{% endfor %}

### Output
{% for i in range(1, completions|length + 1) %}
#### Output for Text {{ i }}
Type: [List of numeric identifiers (or "None") for informativeness type, separated by commas]
Rationale: [Rationale for the identification in short sentences]
Rating: [Rating for text {{ i }}]
Rationale: [Rationale for the rating in short sentences]

{% endfor %}

---

## Annotation

### Input
Instruction: {{ instruction }}

Texts:
{% for completion in completions %}
<text {{ loop.index }}> {{ completion }}
{% endfor %}

### Output
"""


user_prompts: dict[str, str] = {
    "instruction_following": instruction_following_template,
    "honesty": honesty_template,
    "truthfulness": truthfulness_template_without_answer,
    "helpfulness": helpfulness_template_without_answer,
}

system_prompt = """Your role is to evaluate text quality based on given criteria.
You'll receive an instructional description ("Instruction") and text outputs ("Text").
Understand and interpret instructions to evaluate effectively.
Provide annotations for each text with a rating and rationale.
The texts given are independent, and should be evaluated separately."""


def parser(responses: str, aspect: str):
    try:
        responses = responses.split("\n\n")
    except Exception:
        breakpoint()
    annotation = []

    try:
        if aspect in ["instruction_following", "honesty"]:
            pattern = r"Rating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                if matches:
                    rating_search = re.findall(r"\b\d+\b", matches.group(1))
                    rating = rating_search[0] if len(rating_search) > 0 else "1"
                    annotation.append(
                        {
                            "Rating": (rating if matches.group(1) != "N/A" else "N/A"),
                            "Rationale": matches.group(2),
                        }
                    )
                else:
                    annotation.append({"Rating": "1", "Rationale": ""})
        elif aspect in ["truthfulness", "helpfulness"]:
            pattern = r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                if matches:
                    rating_search = re.findall(r"\b\d+\b", matches.group(3))
                    rating = rating_search[0] if len(rating_search) > 0 else "1"
                    annotation.append(
                        {
                            "Type": (
                                re.findall(r"\b\d+\b", matches.group(1))
                                if matches.group(1) != "None"
                                else "None"
                            ),
                            "Rationale": matches.group(2),
                            "Rating": rating,
                            "Rationale For Rating": matches.group(4),
                        }
                    )
                else:
                    annotation.append(
                        {
                            "Type": "None",
                            "Rationale": "None",
                            "Rating": "1",
                            "Rationale For Rating": "None",
                        }
                    )

    except ValueError as e:
        logging.warning(f"Response didn't follow format: {e}")
    except AttributeError as e:
        logging.warning(f"Potential missing keys: {e}")
    return annotation
