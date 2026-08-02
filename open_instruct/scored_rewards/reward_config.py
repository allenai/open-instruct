"""Where a scorer meets open-instruct's reward path.

Two ways in, and you almost always want the second.

``ScoreVerifier``     a per-sample ``Scorer`` exposed as an ordinary
                      ``VerifierFunction``, routed by the dataset's ``dataset``
                      column like any other verifier. Continuous scores already
                      work upstream (``VerificationResult.score`` is a float),
                      so this only exists to spare you editing
                      ``ground_truth_utils.py``.

``GroupRewardConfig`` a drop-in ``RewardConfig`` that additionally runs a
                      ``GroupScorer`` over the whole group.

The group hook needs no new plumbing, because the group is already there.
``vllm_utils.compute_rewards`` is called once per prompt with every sample of
that prompt in ``result.responses`` - open-instruct just hands them to the
verifiers one at a time. This subclass hands them over together.

EVERYTHING IS BUILT FROM A SPEC STRING, LAZILY. The reward config is pickled and
shipped to the vLLM Ray actors, and a scorer holding an open HTTP client, a
tokenizer or a CUDA tensor does not survive that trip. So the config carries the
plugin list and the spec, and the scorer is constructed inside ``build()``,
which runs in the process that will actually call it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

import numpy as np

from open_instruct.ground_truth_utils import RewardConfig, VerificationResult, VerifierConfig, VerifierFunction
from open_instruct.scored_rewards import registry
from open_instruct.scored_rewards.aggregate import is_degenerate
from open_instruct.scored_rewards.types import GroupScorer, Sample, Scorer

logger = logging.getLogger(__name__)

#: ``build_all_verifiers`` instantiates every ``VerifierFunction`` subclass with
#: no scorer attached. That placeholder is dropped in ``make_reward_config``.
PLACEHOLDER_NAME = "scored_reward_unconfigured"


class ScoreVerifier(VerifierFunction):
    """Expose a per-sample ``Scorer`` as an open-instruct verifier.

    Rows route to it by their ``dataset`` column, and the score is multiplied by
    ``--verification_reward`` (10.0 by default) like every other verifier, so
    keep your scorer in [0, 1] unless you mean otherwise.
    """

    def __init__(
        self,
        verifier_config: VerifierConfig | None = None,
        spec: str | None = None,
        plugins: str | None = None,
        name: str | None = None,
        weight: float = 1.0,
    ):
        super().__init__(name=name or PLACEHOLDER_NAME, weight=weight, verifier_config=verifier_config)
        self.spec = spec
        self.plugins = plugins
        self._scorer: Scorer | None = None

    @property
    def scorer(self) -> Scorer | None:
        if self._scorer is None and self.spec:
            registry.load_plugins(self.plugins)
            built = registry.build(self.spec)
            inner = getattr(built, "scorer", built)
            if not isinstance(inner, Scorer):
                raise TypeError(f"{self.spec!r} is a group scorer; use --group_scorer, not --score_verifiers")
            self._scorer = inner
        return self._scorer

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        raise RuntimeError("ScoreVerifier is async-only; async_call is the entry point")

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        scorer = self.scorer
        if scorer is None:
            return VerificationResult(score=0.0, reasoning="ScoreVerifier has no scorer attached")
        sample = Sample(
            completion=prediction,
            prompt=query or "",
            label=label,
            token_ids=list(tokenized_prediction or []),
            rollout=rollout_state or {},
        )
        result = await scorer.score(sample)
        return VerificationResult(score=float(result.score), reasoning=str(result.info) if result.info else None)

    def __reduce__(self):
        # never ship the built scorer across the wire
        return (_rebuild_score_verifier, (self.verifier_config, self.spec, self.plugins, self.name, self.weight))


def _rebuild_score_verifier(verifier_config, spec, plugins, name, weight):
    return ScoreVerifier(verifier_config=verifier_config, spec=spec, plugins=plugins, name=name, weight=weight)


@dataclasses.dataclass
class GroupRewardConfig(RewardConfig):
    """``RewardConfig`` plus a group-aware scoring pass."""

    group_scorer_spec: str | None = None
    reward_plugins: str | None = None
    group_reward_mode: Literal["replace", "add"] = "replace"
    """``replace`` ignores the verifier score, ``add`` sums the two. Replace is
    the default because a group scorer that normalises within the group produces
    a zero-mean quantity, and adding an absolute verifier score to it would
    re-introduce exactly the scale problem the normalisation removed."""
    group_reward_scale: float = 1.0
    group_scorer_strict: bool = False
    """Raise on a scorer error instead of falling back to the verifier score. Off
    by default: one flaky judge call should not end a 250-step run."""
    group_scorer: GroupScorer | None = None
    """Pre-built scorer. For tests; a real run passes ``group_scorer_spec`` so the
    scorer is constructed inside the actor rather than pickled into it."""

    def build(self):
        base_fn = super().build()
        scorer = self.group_scorer
        if scorer is None and self.group_scorer_spec:
            registry.load_plugins(self.reward_plugins)
            scorer = registry.build(self.group_scorer_spec)
            logger.info("scored_rewards: built group scorer %r from %r", scorer, self.group_scorer_spec)
        if scorer is None:
            return base_fn

        async def reward_fn(
            responses: list,
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos,
            queries: list[str] | None = None,
        ) -> tuple[list[float], dict[str, Any]]:
            scores, metrics = await base_fn(
                responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, queries
            )
            group = build_group(responses, decoded_responses, ground_truths, infos, queries)

            try:
                results = await scorer.score_group(group)
            except Exception:
                if self.group_scorer_strict:
                    raise
                logger.exception("scored_rewards: group scorer %r failed; falling back to verifier scores", scorer)
                metrics["scored/group_scorer_failures"] = 1.0
                return scores, metrics

            if len(results) != len(scores):
                raise ValueError(f"group scorer returned {len(results)} results for {len(scores)} samples")

            group_scores = [self.group_reward_scale * float(r.score) for r in results]
            final = [a + b for a, b in zip(scores, group_scores)] if self.group_reward_mode == "add" else group_scores
            metrics.update(summarize(results, group_scores, scorer.name or "group"))
            return final, metrics

        return reward_fn


def build_group(responses, decoded_responses, ground_truths, infos, queries) -> list[Sample]:
    rollout_states = getattr(infos, "rollout_states", None) or [{}] * len(decoded_responses)
    queries = queries or [""] * len(decoded_responses)
    return [
        Sample(
            completion=decoded_responses[i],
            prompt=queries[i] or "",
            label=ground_truths[i],
            token_ids=list(responses[i]) if i < len(responses) else [],
            rollout=rollout_states[i] or {},
            index=i,
            group_size=len(decoded_responses),
        )
        for i in range(len(decoded_responses))
    ]


def summarize(results, group_scores: list[float], name: str) -> dict[str, Any]:
    """Mean of every numeric diagnostic the scorer attached, plus the one metric
    that decides whether the reward still resolves differences at all."""
    prefix = f"scored/{name}"
    metrics: dict[str, Any] = {
        f"{prefix}/reward": float(np.mean(group_scores)) if group_scores else 0.0,
        f"{prefix}/zero_advantage": float(is_degenerate(group_scores)),
    }
    numeric: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.info.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
        for key, value in result.dimensions.items():
            if value is not None:
                numeric.setdefault(f"dim_{key}", []).append(float(value))
    for key, values in numeric.items():
        metrics[f"{prefix}/{key}"] = float(np.mean(values))
    return metrics


def make_reward_config(args, streaming_config, tools_config) -> RewardConfig:
    """Build the reward config, honouring ``--reward_plugins`` and friends.

    Drop-in for the ``RewardConfig(...)`` literal in ``grpo_fast.main``. With
    none of the new flags set it returns exactly what upstream would have built.
    """
    from open_instruct.ground_truth_utils import build_all_verifiers  # noqa: PLC0415

    plugins = getattr(streaming_config, "reward_plugins", None)
    registry.load_plugins(plugins)

    verifiers = build_all_verifiers(args, streaming_config)
    verifiers.pop(PLACEHOLDER_NAME, None)
    for spec in _split(getattr(streaming_config, "score_verifiers", None)):
        name = spec.partition(":")[0].strip().lower()
        verifiers[name] = ScoreVerifier(spec=spec, plugins=plugins, name=name)

    common = dict(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=verifiers,
        reward_aggregator=streaming_config.reward_aggregator,
    )

    spec = getattr(streaming_config, "group_scorer", None)
    if not spec:
        return RewardConfig(**common)

    # fail here, in the launcher, rather than inside a Ray actor twenty minutes in
    registry.build(spec)
    return GroupRewardConfig(
        **common,
        group_scorer_spec=spec,
        reward_plugins=plugins,
        group_reward_mode=getattr(streaming_config, "group_reward_mode", "replace"),
        group_reward_scale=getattr(streaming_config, "group_reward_scale", 1.0),
        group_scorer_strict=getattr(streaming_config, "group_scorer_strict", False),
    )


def _split(spec: str | None) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]
