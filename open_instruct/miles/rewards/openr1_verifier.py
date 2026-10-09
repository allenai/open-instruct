"""Grade OpenR1 answers with Math-Verify's LaTeX extraction and equivalence rules."""

import importlib

from open_instruct.miles.rewards import task_verifiers


class OpenR1MathVerifier:
    """Compare raw OpenR1 answer targets against anchored model answers."""

    @classmethod
    def get_config_class(cls):
        return task_verifiers.RewardConfig

    def __init__(self, verifier_config=None):
        # Keep optional symbolic dependencies inside the isolated grading worker.
        self.math = importlib.import_module("math_verify")
        normalization = importlib.import_module("latex2sympy2_extended").NormalizationConfig
        self.prediction_config = self.math.LatexExtractionConfig(
            normalization_config=normalization(
                nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )

    def __call__(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("OpenR1 grading requires a nonempty answer target")
        # Dataset answers are bare LaTeX, unlike generated answers with delimiters.
        gold = self.math.parse(
            "\\boxed{" + label + "}",
            extraction_config=[self.math.LatexExtractionConfig()],
            extraction_mode="first_match",
        )
        if not gold:
            raise ValueError("OpenR1 answer target could not be parsed")
        answer = self.math.parse(prediction, extraction_config=[self.prediction_config], extraction_mode="first_match")
        return task_verifiers.RewardResult(float(self.math.verify(gold, answer)))
