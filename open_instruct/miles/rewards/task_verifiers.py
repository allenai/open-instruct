"""Provide lightweight reward adapters for prepared task datasets. Multiplication
and answer-format checks run locally, while the instruction-following adapter
translates manifest targets into Open Instruct's existing verifier format. Shared
answer normalization and deferred grading imports let data preparation validate
these targets without loading the full math and NLP dependency stack.
"""

import ast
import dataclasses
import importlib
import json
import re

from open_instruct.miles.errors import InputError


def scalar_target(target):
    if isinstance(target, list):
        if len(target) != 1:
            raise InputError("Expected a singleton answer")
        target = target[0]
    if target is None or not str(target).strip() or isinstance(target, dict):
        raise InputError("Expected a nonempty scalar answer")
    normalized = str(target).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        raise InputError("Scalar answer must not be a stringified list")
    return normalized.replace(",", "")


@dataclasses.dataclass
class RewardConfig:
    pass


@dataclasses.dataclass
class RewardResult:
    score: float
    cost: float = 0.0


class MultiplicationVerifier:
    """Preserve the baseline's answer-tag numeric scorer."""

    def __init__(self, verifier_config=None):
        pass

    @classmethod
    def get_config_class(cls):
        return RewardConfig

    async def async_call(self, tokens, prediction, label, **kwargs):
        try:
            answer = prediction[prediction.find("<answer>") + len("<answer>") : prediction.find("</answer>")]
            score = float(float(answer.replace(",", "").strip()) == float(scalar_target(label)))
        except (TypeError, ValueError):
            score = 0.0
        return RewardResult(score)


class R1FormatVerifier(MultiplicationVerifier):
    async def async_call(self, tokens, prediction, label, **kwargs):
        return RewardResult(float(re.match(r".*?</think>\s*<answer>.*?</answer>", prediction, re.DOTALL) is not None))


class ManifestIFVerifier(MultiplicationVerifier):
    """Translate canonical baseline constraint targets to OI's legacy list wrapper."""

    async def async_call(self, tokens, prediction, label, **kwargs):
        target = label
        if isinstance(target, str):
            try:
                target = json.loads(target)
            except ValueError:
                target = ast.literal_eval(target)
        if isinstance(target, list):
            if len(target) != 1:
                raise InputError("IF target requires exactly one constraint bundle")
            target = target[0]
            if isinstance(target, str):
                target = json.loads(target)
        if not isinstance(target, dict) or "instruction_id" not in target or "kwargs" not in target:
            raise InputError("IF target requires instruction_id and kwargs")
        # Grading imports math/NLP dependencies; defer them so dataset planning
        # and the lightweight preparation verifiers work without that stack.
        factory = importlib.import_module("open_instruct.ground_truth_utils").IFEvalVerifier
        return await factory().async_call(tokens, prediction, repr([target]), **kwargs)


@dataclasses.dataclass
class QualifiedIFConfig:
    module: str = "experimental_ifeval_reward_v2"


class QualifiedIFVerifier:
    """Use a qualified IF module's raw-response or extracted-answer entry point.

    The module retains ownership of its target validation, constraint grading and
    dependency pins. Raw responses still use its IFVerifier unchanged. In final-
    answer-only mode the reward adapter owns the boundary gate, so grade_answer
    must receive the selected suffix without a fabricated closing marker.
    """

    def __init__(self, verifier_config=None):
        config = verifier_config or QualifiedIFConfig()
        module = importlib.import_module(config.module)
        self.verifier = module.IFVerifier(verifier_config=module.IFVerifier.get_config_class()())
        self.grade_answer = module.grade_answer

    @classmethod
    def get_config_class(cls):
        return QualifiedIFConfig

    async def async_call(self, tokens, prediction, label, **kwargs):
        return await self.verifier.async_call(tokens, prediction, label, **kwargs)

    async def async_call_final_answer(self, tokens, prediction, label, **kwargs):
        return self.grade_answer(prediction, label)
