"""Drives the REAL upstream seam, not a stand-in.

``test_scored_rewards.py`` tests this package in isolation and runs anywhere.
This file is the other half: it builds a genuine
``ground_truth_utils.RewardConfig``, calls ``.build()``, and invokes the
returned ``reward_fn`` with exactly the arguments
``vllm_utils.compute_rewards`` passes it. That is the only check that catches
the failure mode that matters on a rebase - upstream changing the reward
signature or the ``RewardConfig`` fields underneath us.

It needs open-instruct's own dependencies but still no GPU, no ray cluster, no
vLLM and no network. Where those are missing the whole module skips, so a laptop
run of the other file stays green.

    python -m unittest open_instruct.scored_rewards.test_integration -v

Inside the project environment this just works. To run it standalone - which is
enough to check the seam after a rebase, and does not need vLLM - the import
chain bottoms out at:

    pip install numpy requests tiktoken transformers absl-py sympy \
                nltk immutabledict langdetect beaker ray torch wandb datasets
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import unittest

try:
    from open_instruct.ground_truth_utils import RewardConfig  # noqa: F401
    from open_instruct.scored_rewards.reward_config import GroupRewardConfig, make_reward_config

    DEPS = True
except Exception as exc:  # pragma: no cover - depends on the install
    DEPS = False
    REASON = f"open-instruct deps not installed ({type(exc).__name__}: {exc})"

from open_instruct.scored_rewards import ScoreResult, registry
from open_instruct.scored_rewards.types import GroupScorer


class LengthScorer(GroupScorer):
    name = "itest_len"

    async def score_group(self, group):
        return [ScoreResult(score=len(s.completion) / 100.0, info={"chars": len(s.completion)}) for s in group]


class FlatScorer(GroupScorer):
    name = "itest_flat"

    async def score_group(self, group):
        return [ScoreResult(score=1.0) for _ in group]


class BoomScorer(GroupScorer):
    name = "itest_boom"

    async def score_group(self, group):
        raise RuntimeError("judge endpoint down")


registry.register("itest_len", lambda **kw: LengthScorer())
registry.register("itest_flat", lambda **kw: FlatScorer())
registry.register("itest_boom", lambda **kw: BoomScorer())


@dataclasses.dataclass
class FakeRequestInfo:
    """The attributes upstream's reward_fn reads off ``result.request_info``."""

    timeouts: list
    tool_errors: list
    tool_outputs: list
    tool_calleds: list
    rollout_states: list


class FakeConfig:
    """Stands in for Args / StreamingDataLoaderConfig / EnvsConfig at once.

    Only the fields the reward path actually reads. If upstream adds one, this
    raises a clear AttributeError rather than drifting silently.
    """

    apply_r1_style_format_reward = False
    r1_style_format_reward = 1.0
    apply_verifiable_reward = True
    verification_reward = 10.0
    non_stop_penalty = False
    non_stop_penalty_value = -10.0
    only_reward_good_outputs = False
    additive_format_reward = False
    reward_aggregator = "last"
    seed = 0

    reward_plugins = None
    group_scorer = None
    group_reward_mode = "replace"
    group_reward_scale = 1.0
    group_scorer_strict = False
    score_verifiers = None

    llm_judge_model = "gpt-4o"
    llm_judge_max_tokens = 2048
    llm_judge_max_context_length = 8192
    llm_judge_temperature = 1.0
    llm_judge_timeout = 60
    code_api_url = "http://localhost:1234/test_program"
    code_max_execution_time = 1.0
    code_pass_rate_reward_threshold = 0.0
    code_apply_perf_penalty = False
    remap_verifier = None
    max_length_verifier_max_length = 32768


COMPLETIONS = ["short", "a bit longer here", "x" * 60, "y" * 120]


def run_reward(**overrides):
    """Build a reward config the way grpo_fast does, then call it the way the
    Ray actor does. Returns (config class name, scores, metrics)."""
    cfg = FakeConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)

    reward_config = make_reward_config(cfg, cfg, cfg)
    reward_fn = reward_config.build()

    k = len(COMPLETIONS)
    info = FakeRequestInfo(
        timeouts=[False] * k,
        tool_errors=[""] * k,
        tool_outputs=[""] * k,
        tool_calleds=[False] * k,
        rollout_states=[{} for _ in range(k)],
    )
    scores, metrics = asyncio.run(
        reward_fn(
            [[1, 2, 3]] * k,
            list(COMPLETIONS),
            [json.dumps({"answer": "copper"})] * k,
            ["passthrough"] * k,
            ["stop"] * k,
            info,
            ["what conducts electricity?"] * k,
        )
    )
    return type(reward_config).__name__, scores, metrics


@unittest.skipUnless(DEPS, "" if DEPS else REASON)
class TestUpstreamSeam(unittest.TestCase):
    def test_no_flags_returns_the_plain_upstream_config(self):
        """The whole promise of the patch: unset flags change nothing."""
        name, scores, _ = run_reward()
        self.assertEqual(name, "RewardConfig")
        self.assertEqual(len(scores), len(COMPLETIONS))

    def test_group_scorer_replaces_the_score(self):
        name, scores, metrics = run_reward(group_scorer="itest_len")
        self.assertEqual(name, GroupRewardConfig.__name__)
        self.assertEqual([round(s, 6) for s in scores], [0.05, 0.17, 0.6, 1.2])
        self.assertIn("scored/itest_len/reward", metrics)

    def test_scorer_info_becomes_a_metric(self):
        _, _, metrics = run_reward(group_scorer="itest_len")
        self.assertAlmostEqual(metrics["scored/itest_len/chars"], 50.5)

    def test_scale_multiplies_the_group_score(self):
        _, scores, _ = run_reward(group_scorer="itest_len", group_reward_scale=10.0)
        self.assertEqual([round(s, 6) for s in scores], [0.5, 1.7, 6.0, 12.0])

    def test_add_mode_keeps_the_verifier_score(self):
        _, replaced, _ = run_reward(group_scorer="itest_len")
        _, added, _ = run_reward(group_scorer="itest_len", group_reward_mode="add")
        for r, a in zip(replaced, added):
            self.assertGreaterEqual(a, r)

    def test_degenerate_group_is_reported(self):
        """All-equal scores mean zero advantage everywhere, which is the signal
        that the reward has stopped resolving differences."""
        _, _, flat = run_reward(group_scorer="itest_flat")
        _, _, spread = run_reward(group_scorer="itest_len")
        self.assertEqual(flat["scored/itest_flat/zero_advantage"], 1.0)
        self.assertEqual(spread["scored/itest_len/zero_advantage"], 0.0)

    def test_a_failing_scorer_does_not_kill_the_run(self):
        _, scores, metrics = run_reward(group_scorer="itest_boom")
        self.assertEqual(len(scores), len(COMPLETIONS))
        self.assertEqual(metrics["scored/itest_boom/failed"], 1.0)

    def test_the_failure_metric_is_present_when_nothing_failed(self):
        """It has to chart as a rate, so it cannot only appear on the bad path."""
        _, _, metrics = run_reward(group_scorer="itest_len")
        self.assertEqual(metrics["scored/itest_len/failed"], 0.0)

    def test_strict_mode_raises(self):
        with self.assertRaises(RuntimeError):
            run_reward(group_scorer="itest_boom", group_scorer_strict=True)

    def test_unknown_scorer_fails_in_the_launcher(self):
        """Not twenty minutes later inside a Ray actor."""
        with self.assertRaises(KeyError):
            run_reward(group_scorer="no_such_scorer")


if __name__ == "__main__":
    unittest.main()
