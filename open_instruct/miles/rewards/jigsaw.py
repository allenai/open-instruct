"""Exact, deliberately narrow answer scoring for the two-choice jigsaw pilot."""

import re

from open_instruct.miles.rewards import task_verifiers


def answer(prediction: str) -> str | None:
    """Accept one choice and an optional terminal Olmo EOS from native rollouts."""
    if not isinstance(prediction, str):
        return None
    match = re.fullmatch(r"\s*(?:Answer:\s*)?(?:([AB])|\(([AB])\))\.?\s*(?:<\|endoftext\|>)?\s*", prediction)
    return (match[1] or match[2]) if match else None


class JigsawVerifier:
    def __init__(self, verifier_config=None):
        pass

    @classmethod
    def get_config_class(cls):
        return task_verifiers.RewardConfig

    async def async_call(self, tokens, prediction, label, **kwargs):
        if label not in ("A", "B"):
            raise ValueError("Jigsaw target must be A or B")
        return task_verifiers.RewardResult(float(answer(prediction) == label))
