"""Math prompt formatting and data checks for the MILES math reward.

MathVerifier reads an answer from ``\\boxed{}``, a "the final answer is X" sentence or
the last math span, so a correct answer written any other way scores zero. Almost
no Dolci math prompt states the expected format, which made learning to box most of
the Phase 1 math gain. ``add_boxed_instruction`` states the format in the prompt;
``math_row_problems`` names source rows whose targets cannot be trusted.
"""

import re

MATH_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# A prompt that already names an answer format keeps its own instruction.
_FORMAT_REQUEST_PATTERN = re.compile(r"\\boxed|\bboxed\b|final\s+answer", re.IGNORECASE)


def add_boxed_instruction(prompt_text: str) -> str:
    """Append the boxed-answer instruction unless the prompt already asks for a format."""
    if _FORMAT_REQUEST_PATTERN.search(prompt_text):
        return prompt_text
    return f"{prompt_text.rstrip()}\n\n{MATH_BOXED_INSTRUCTION}"


def math_row_problems(row: dict) -> list[str]:
    """Reasons a source math row needs review or removal; empty when the target looks usable.

    ``row["ground_truth"]`` is the Open Instruct target, either a string or a
    one-element list of strings. ``placeholder_target`` marks "-1": 168 prepared
    math rows carry it, and the Phase 1 audit could not tell genuine answers from
    placeholders, so these rows need review before they train or evaluate anything.
    """
    target = row.get("ground_truth")
    if isinstance(target, list):
        if len(target) != 1:
            return ["not_single_target"]
        target = target[0]
    if not isinstance(target, str) or not target.strip():
        return ["empty_target"]
    if target.strip() == "-1":
        return ["placeholder_target"]
    return []
