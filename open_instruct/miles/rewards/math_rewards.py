"""The MILES math reward: Math-Verify on the boxed answer, prompt formatting and data checks.

``MathVerifyVerifier`` scores a response with Hugging Face Math-Verify in its strict
configuration: the answer must be in ``\\boxed{}`` (or follow an explicit "final answer"
anchor), and Math-Verify decides symbolic equivalence. There is no bare-number fallback,
so the boxed format is the one format that pays. Almost no Dolci math prompt states that
format, which made learning to box most of the Phase 1 math gain, so
``add_boxed_instruction`` states it in the prompt. ``math_row_problems`` names source
rows whose targets cannot be trusted.

The shared ``ground_truth_utils.MathVerifier`` is unchanged, so the vLLM GRPO trainers
keep their rewards.
"""

import dataclasses
import re

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from open_instruct.ground_truth_utils import (
    VerificationResult,
    VerifierConfig,
    VerifierFunction,
    remove_thinking_section,
)

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


# Targets are bare LaTeX ("\\frac{1}{2}", " n = 5 "), so they parse as one math span.
_GOLD_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]
# open-r1's strict response extraction: boxed or anchored LaTeX only, no bare expressions.
_ANSWER_EXTRACTION = [LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False)]


@dataclasses.dataclass
class MathVerifyConfig(VerifierConfig):
    parsing_timeout_seconds: int = 5
    verify_timeout_seconds: int = 5


class MathVerifyVerifier(VerifierFunction):
    """Math-Verify equivalence on the response's boxed (or anchored) answer.

    Math-Verify's timeouts use SIGALRM, so the MILES reward bridge runs this verifier in
    its main-thread math worker process, like ``MathVerifier``.
    """

    def __init__(self, verifier_config: MathVerifyConfig | None = None) -> None:
        self.config = verifier_config or MathVerifyConfig()
        super().__init__("math_verify", weight=1.0, verifier_config=self.config)

    @classmethod
    def get_config_class(cls) -> type:
        return MathVerifyConfig

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | list[str],
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        answer = parse(
            remove_thinking_section(prediction),
            extraction_config=_ANSWER_EXTRACTION,
            parsing_timeout=self.config.parsing_timeout_seconds,
        )
        if not answer:
            return VerificationResult(score=0.0)
        for target in label if isinstance(label, list) else [label]:
            gold = parse(
                f"${target}$", extraction_config=_GOLD_EXTRACTION, parsing_timeout=self.config.parsing_timeout_seconds
            )
            if gold and verify(gold, answer, timeout_seconds=self.config.verify_timeout_seconds):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)
