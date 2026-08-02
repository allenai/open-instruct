"""The seams.

Pure types plus two small base classes. Nothing here imports torch, vLLM, ray or
openenv, so a scorer can be written and unit-tested without the training stack
installed.

Two levels exist because GRPO has two levels.

``Scorer``       looks at one completion. This is the familiar reward-model
                 shape and covers most cases.
``GroupScorer``  looks at all G completions sampled from one prompt at once.
                 open-instruct already computes rewards a whole group at a time
                 (``vllm_utils.compute_rewards`` is called once per prompt with
                 every sample in ``result.responses``), so the group is free to
                 ask for. It is what you need for any reward that is relative
                 rather than absolute: per-dimension normalisation, a
                 counterfactual baseline drawn from a sibling, or a rank.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import json
from typing import Any


@dataclasses.dataclass
class Sample:
    """One completion, with everything a scorer might need to judge it.

    ``completion`` is the decoded policy output. In a multi-turn rollout it is
    the WHOLE stream, environment turns included, because that is what the model
    generated into. ``policy_text`` holds the policy's turns alone when the
    environment recorded them (see ``partner_env``); reward rules that punish
    the policy for something it said must read that one, or they will charge it
    for words the environment put in its mouth.
    """

    completion: str
    prompt: str = ""
    label: Any = None
    token_ids: list[int] = dataclasses.field(default_factory=list)
    rollout: dict = dataclasses.field(default_factory=dict)
    index: int = 0
    group_size: int = 1

    @property
    def item(self) -> dict:
        """``label`` as a dict.

        The dataset's ``ground_truth`` column is a string, so an item with any
        structure to it (a question, its options, a rubric, a reference) travels
        as JSON. Anything that is not a JSON object comes back under ``answer``
        so a scorer can read one key either way.
        """
        if isinstance(self.label, dict):
            return self.label
        if isinstance(self.label, str):
            try:
                parsed = json.loads(self.label)
            except (json.JSONDecodeError, ValueError):
                return {"answer": self.label}
            return parsed if isinstance(parsed, dict) else {"answer": parsed}
        return {"answer": self.label}

    @property
    def env_info(self) -> dict:
        """Whatever the environment recorded, under its namespace."""
        return (self.rollout or {}).get("info", {}) or {}

    @property
    def policy_text(self) -> str:
        """The policy's own turns, if the environment separated them out."""
        return self.env_info.get(POLICY_TEXT_KEY) or self.completion

    @property
    def turn_rewards(self) -> list[float]:
        return list((self.rollout or {}).get("rewards", []))


#: Key an environment writes its policy-only transcript under, in ``get_metrics``.
POLICY_TEXT_KEY = "scored_policy_text"
#: Key an environment writes its ``[{"who": ..., "text": ...}]`` transcript under.
TRANSCRIPT_KEY = "scored_transcript"


def parse_transcript(text: str) -> list[dict]:
    """Read back the transcript an environment recorded under ``TRANSCRIPT_KEY``."""
    try:
        blob = json.loads(text or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return blob if isinstance(blob, list) else []


@dataclasses.dataclass
class ScoreResult:
    """What a scorer returns.

    ``score`` is the scalar the RL update consumes. ``dimensions`` is optional
    and only meaningful to a multi-dimensional aggregator: if it is populated,
    ``MultiDimensionalScorer`` normalises each dimension inside the group and
    ignores ``score``. ``info`` is never optimised - it is logged, so that a
    suspicious reward can be read rather than guessed at.

    A dimension the scorer could not produce is ``None``, not a middling
    default. Under group normalisation a systematic pull toward the mean is a
    bias, not noise, and the caller should prefer to drop the sample.
    """

    score: float = 0.0
    dimensions: dict[str, float | None] = dataclasses.field(default_factory=dict)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class Scorer(abc.ABC):  # noqa: B024 - either method may be the override, so neither can be abstract
    """Scores one completion.

    Override ``score`` for an async scorer (a judge behind an HTTP endpoint) or
    ``score_sync`` for a plain one (a string rule, a local head). The default
    ``score`` runs ``score_sync`` in a thread so a blocking scorer cannot stall
    the actor's event loop, which mirrors what open-instruct's own
    ``VerifierFunction.async_call`` does.
    """

    name: str = ""

    async def score(self, sample: Sample) -> ScoreResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score_sync, sample)

    def score_sync(self, sample: Sample) -> ScoreResult:
        raise NotImplementedError(f"{type(self).__name__} implements neither score nor score_sync")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class GroupScorer(abc.ABC):
    """Scores all G completions of one prompt together."""

    name: str = ""

    @abc.abstractmethod
    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        """Return one result per sample, in the order given."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class PerSample(GroupScorer):
    """Lift a ``Scorer`` into a ``GroupScorer``, scoring the group in parallel."""

    def __init__(self, scorer: Scorer, name: str | None = None):
        self.scorer = scorer
        self.name = name or scorer.name or type(scorer).__name__

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        return list(await asyncio.gather(*(self.scorer.score(s) for s in group)))


class FunctionScorer(Scorer):
    """Wrap a plain callable ``(Sample) -> float | dict | ScoreResult``.

    The escape hatch for a reward you can write in ten lines and do not want to
    make a class for.
    """

    def __init__(self, fn, name: str = ""):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")

    def score_sync(self, sample: Sample) -> ScoreResult:
        return as_result(self.fn(sample))


def as_result(value: Any) -> ScoreResult:
    """Coerce whatever a user's scorer returned into a ``ScoreResult``."""
    if isinstance(value, ScoreResult):
        return value
    if isinstance(value, dict):
        info = {k: v for k, v in value.items() if k not in ("score", "dimensions", "info")}
        info.update(value.get("info", {}))
        return ScoreResult(
            score=float(value.get("score", 0.0)), dimensions=dict(value.get("dimensions", {})), info=info
        )
    return ScoreResult(score=float(value))
