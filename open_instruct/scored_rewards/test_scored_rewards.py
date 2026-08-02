"""Unit tests for the generic layer.

These deliberately need nothing installed beyond the standard library - no
torch, vllm, ray or openenv - because the whole point of the seams is that a
reward can be written and checked on a laptop before it costs a GPU hour.

    python -m unittest open_instruct.scored_rewards.test_scored_rewards -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

from open_instruct.scored_rewards import aggregate, anchor, data, guards, judge, registry
from open_instruct.scored_rewards.types import (
    FunctionScorer,
    GroupScorer,
    PerSample,
    Sample,
    Scorer,
    ScoreResult,
    parse_transcript,
)


def run(coro):
    return asyncio.run(coro)


class TestSample(unittest.TestCase):
    def test_json_label_becomes_item(self):
        payload = {"question": "why?", "choices": ["a", "b"], "gold_idx": 1}
        sample = Sample(completion="hi", label=json.dumps(payload))
        self.assertEqual(sample.item, payload)

    def test_plain_label_lands_under_answer(self):
        self.assertEqual(Sample(completion="", label="copper wire").item, {"answer": "copper wire"})
        # a bare number is valid JSON, so it arrives parsed
        self.assertEqual(Sample(completion="", label="42").item, {"answer": 42})

    def test_malformed_json_does_not_raise(self):
        self.assertEqual(Sample(completion="", label="{not json").item, {"answer": "{not json"})

    def test_policy_text_falls_back_to_completion(self):
        sample = Sample(completion="everything")
        self.assertEqual(sample.policy_text, "everything")

    def test_policy_text_prefers_what_the_env_recorded(self):
        sample = Sample(completion="tutor and student", rollout={"info": {"scored_policy_text": "tutor only"}})
        self.assertEqual(sample.policy_text, "tutor only")

    def test_parse_transcript_tolerates_garbage(self):
        self.assertEqual(parse_transcript("nope"), [])
        self.assertEqual(parse_transcript('[{"who": "policy", "text": "x"}]'), [{"who": "policy", "text": "x"}])


class TestAggregate(unittest.TestCase):
    def test_normalize_then_sum_is_zero_mean(self):
        rows = [{"a": 0.1, "b": 0.9}, {"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}]
        out = aggregate.normalize_then_sum(rows, ("a", "b"))
        self.assertAlmostEqual(sum(out), 0.0, places=9)

    def test_opposed_dimensions_cancel(self):
        """a and b are perfectly anticorrelated, so nothing is preferred."""
        rows = [{"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}]
        out = aggregate.normalize_then_sum(rows, ("a", "b"))
        self.assertTrue(all(abs(v) < 1e-9 for v in out), out)

    def test_scale_invariance(self):
        """Ten-fold one dimension and the result barely moves.

        This is the property the whole ordering exists for: with a single summed
        reward, scaling `a` by ten would hand it the entire gradient. Invariance
        is exact up to the epsilon in the denominator, which is why this is a
        tolerance and not an equality.
        """
        small = [{"a": 0.1, "b": 0.2}, {"a": 0.2, "b": 0.9}, {"a": 0.3, "b": 0.4}]
        large = [{"a": 1.0, "b": 0.2}, {"a": 2.0, "b": 0.9}, {"a": 3.0, "b": 0.4}]
        for x, y in zip(
            aggregate.normalize_then_sum(small, ("a", "b")), aggregate.normalize_then_sum(large, ("a", "b"))
        ):
            self.assertAlmostEqual(x, y, delta=1e-2)

    def test_flat_dimension_adds_zero_and_dilutes(self):
        """A dimension with no spread does not vote - but it is still counted in
        the denominator, so it halves the magnitude. See the note in aggregate."""
        varying = [{"a": 0.1, "b": 0.5}, {"a": 0.9, "b": 0.5}]
        alone = [{"a": 0.1}, {"a": 0.9}]
        with_flat = aggregate.normalize_then_sum(varying, ("a", "b"))
        without = aggregate.normalize_then_sum(alone, ("a",))
        for x, y in zip(with_flat, without):
            self.assertAlmostEqual(x, y / 2, places=9)

    def test_missing_is_treated_as_the_mean(self):
        rows = [{"a": 0.0}, {"a": 1.0}, {"a": None}]
        out = aggregate.normalize_then_sum(rows, ("a",))
        self.assertAlmostEqual(out[2], 0.0, places=6)
        self.assertEqual(aggregate.count_missing(rows, ("a",)), 1)

    def test_degenerate_group(self):
        self.assertTrue(aggregate.is_degenerate([0.3, 0.3, 0.3]))
        self.assertFalse(aggregate.is_degenerate([0.3, 0.4]))
        self.assertEqual(aggregate.zero_advantage_fraction([[1.0, 1.0], [0.0, 1.0]]), 0.5)


class Constant(Scorer):
    def __init__(self, value=1.0, dimensions=None, name="constant"):
        self.value, self._dimensions, self.name = value, dimensions or {}, name

    def score_sync(self, sample):
        return ScoreResult(score=self.value, dimensions=dict(self._dimensions))


class ByItem(Scorer):
    """Scores 1.0 only when the completion mentions the item's keyword."""

    name = "by_item"

    def score_sync(self, sample):
        keyword = sample.item.get("keyword", "")
        return ScoreResult(score=1.0 if keyword and keyword in sample.completion else 0.0)


class TestGuards(unittest.TestCase):
    def test_veto_overrides_and_logs(self):
        scorer = guards.Veto(Constant(1.0, {"q": 1.0}), rule=lambda s: "bad" in s.completion, floor=-1.0, name="leak")
        clean = run(scorer.score(Sample(completion="fine")))
        dirty = run(scorer.score(Sample(completion="bad")))
        self.assertEqual(clean.score, 1.0)
        self.assertEqual(dirty.score, -1.0)
        self.assertEqual(dirty.dimensions["q"], -1.0)
        self.assertEqual(dirty.info["leak_fired"], 1.0)
        self.assertEqual(clean.info["leak_fired"], 0.0)

    def test_gate_zeroes_rather_than_penalising(self):
        scorer = guards.Gate(Constant(0.8), predicate=lambda s, r: len(s.completion) > 3)
        self.assertEqual(run(scorer.score(Sample(completion="ab"))).score, 0.0)
        self.assertEqual(run(scorer.score(Sample(completion="abcd"))).score, 0.8)

    def _group(self, completions, keyword):
        """A realistic GRPO group: G samples of ONE prompt, so one shared item."""
        label = json.dumps({"question": "q1", "keyword": keyword})
        return [
            Sample(completion=c, label=label, index=i, group_size=len(completions)) for i, c in enumerate(completions)
        ]

    def test_contrast_pays_for_item_specific_text(self):
        pool = guards.ItemPool([{"question": "q2", "keyword": "beta"}])
        group = self._group(["alpha here", "nothing useful"], keyword="alpha")
        results = run(guards.Contrast(ByItem(), pool).score_group(group))
        self.assertEqual(results[0].score, 1.0)  # works on its own item only
        self.assertEqual(results[1].score, 0.0)  # works on neither
        self.assertEqual(results[0].info["contrast_off"], 0.0)

    def test_contrast_gives_nothing_for_generic_text(self):
        """Text that works on every item earns zero, which is the whole idea."""
        pool = guards.ItemPool([{"question": "q2", "keyword": "beta"}])
        group = self._group(["alpha and beta", "alpha and beta"], keyword="alpha")
        results = run(guards.Contrast(ByItem(), pool).score_group(group))
        self.assertEqual([r.score for r in results], [0.0, 0.0])

    def test_item_pool_never_returns_the_groups_own_item(self):
        own = {"question": "q1", "keyword": "alpha"}
        pool = guards.ItemPool([own, {"question": "q2", "keyword": "beta"}])
        group = self._group(["x"], keyword="alpha")
        self.assertEqual(pool(group)["question"], "q2")

    def test_item_pool_is_deterministic_and_shared_across_the_group(self):
        pool = guards.ItemPool([{"question": f"q{i}", "keyword": str(i)} for i in range(5)])
        group = self._group(["a", "b", "c"], keyword="alpha")
        self.assertEqual(pool(group), pool(group))

    def test_multi_dimensional_normalises_within_the_group(self):
        group = [Sample(completion="a", index=i, group_size=3) for i in range(3)]

        class Varying(GroupScorer):
            name = "varying"

            async def score_group(self, group):
                values = [{"p": 0.1, "q": 0.9}, {"p": 0.5, "q": 0.5}, {"p": 0.9, "q": 0.1}]
                return [ScoreResult(dimensions=v) for v in values]

        results = run(guards.MultiDimensional(Varying(), ("p", "q")).score_group(group))
        self.assertAlmostEqual(sum(r.score for r in results), 0.0, places=9)
        self.assertEqual(results[0].info["dimensions_missing"], 0.0)

    def test_weighted_sums_parts(self):
        parts = {"a": (PerSample(Constant(1.0)), 2.0), "b": (PerSample(Constant(0.5)), 1.0)}
        results = run(guards.Weighted(parts).score_group([Sample(completion="x")]))
        self.assertAlmostEqual(results[0].score, 2.5)


class TestRegistry(unittest.TestCase):
    def test_register_build_and_kwargs(self):
        registry.register("unit_test_scorer", lambda value=0.0: Constant(float(value)))
        built = registry.build("unit_test_scorer:value=0.25")
        self.assertIsInstance(built, PerSample)
        self.assertEqual(run(built.score_group([Sample(completion="")]))[0].score, 0.25)

    def test_parse_kwargs_handles_json_and_commas(self):
        parsed = registry.parse_kwargs("a=1,b=hello,c=[1,2],d=true")
        self.assertEqual(parsed, {"a": 1, "b": "hello", "c": [1, 2], "d": True})

    def test_unknown_name_says_what_to_do(self):
        with self.assertRaises(KeyError) as caught:
            registry.build("definitely_not_registered")
        self.assertIn("reward_plugins", str(caught.exception))

    def test_register_fn_wraps_a_plain_callable(self):
        registry.register_fn("unit_test_fn")(lambda sample: 0.75)
        self.assertEqual(run(registry.build("unit_test_fn").score_group([Sample(completion="")]))[0].score, 0.75)

    def test_load_plugin_from_a_file_path(self):
        source = (
            "from open_instruct.scored_rewards import registry\n"
            "registry.register_fn('unit_test_from_file')(lambda s: 0.5)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "my_plugin.py")
            with open(path, "w") as f:
                f.write(source)
            registry.load_plugins(path)
        self.assertIn("unit_test_from_file", registry.available())


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.rubric = judge.Rubric(
            dimensions=(
                judge.Dimension("clarity", "Is it clear?", {1: "no", 5: "yes"}),
                judge.Dimension("depth", "Does it go deep?", {1: "no", 5: "yes"}),
            )
        )

    def test_prompt_contains_the_schema_and_anchors(self):
        prompt = self.rubric.prompt("SOME WORK")
        self.assertIn("SOME WORK", prompt)
        self.assertIn('"clarity": {"why": "...", "score": N}', prompt)
        self.assertIn("1=no", prompt)

    def test_scores_normalise_to_unit_interval(self):
        reply = 'Here you go: {"clarity": {"why": "x", "score": 5}, "depth": {"why": "y", "score": 1}}'
        parsed = judge.parse_scores(reply, self.rubric)
        self.assertEqual(parsed, {"clarity": 1.0, "depth": 0.0})

    def test_out_of_range_and_missing_become_none(self):
        parsed = judge.parse_scores('{"clarity": {"score": 9}}', self.rubric)
        self.assertIsNone(parsed["clarity"])
        self.assertIsNone(parsed["depth"])

    def test_unparseable_reply_is_all_none_not_all_average(self):
        parsed = judge.parse_scores("I refuse to answer.", self.rubric)
        self.assertEqual(set(parsed.values()), {None})

    def test_bare_number_form_also_parses(self):
        self.assertEqual(judge.parse_scores('{"clarity": 3, "depth": 3}', self.rubric)["clarity"], 0.5)

    def test_reasons_are_extracted_separately(self):
        reasons = judge.parse_reasons('{"clarity": {"why": "vague", "score": 2}}', ["clarity"])
        self.assertEqual(reasons, {"clarity": "vague"})

    def test_judge_counts_parse_failures(self):
        async def broken(prompts):
            return ["nonsense"] * len(prompts)

        j = judge.Judge(broken, self.rubric)
        scores, _ = run(j.score(["a", "b"]))
        self.assertEqual(j.parse_failures, 2)
        self.assertEqual(len(scores), 2)

    def test_stub_generator_round_trips(self):
        j = judge.Judge(judge.stub_generator(self.rubric), self.rubric)
        scores, _ = run(j.score(["something to grade"]))
        self.assertEqual(j.parse_failures, 0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores[0].values()))

    def test_mean_over_turns_ignores_unparsed_turns(self):
        turns = [{"clarity": 1.0}, {"clarity": None}, {"clarity": 0.0}]
        self.assertEqual(judge.mean_over_turns(turns, ["clarity"])["clarity"], 0.5)

    def test_rubric_from_dict(self):
        loaded = judge.Rubric.from_dict(
            {"dimensions": {"tone": {"question": "Is the tone right?", "anchors": {"1": "no", "5": "yes"}}}}
        )
        self.assertEqual(loaded.names, ("tone",))


class TestData(unittest.TestCase):
    def test_build_rows_shape(self):
        items = [{"question": "why?", "choices": ["a", "b"], "gold_idx": 0}]
        rows = data.build_rows(items, system="be nice", user=lambda i: i["question"], env_name="partner_model")
        row = rows[0]
        self.assertEqual(row["dataset"], "passthrough")
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user"])
        self.assertEqual(json.loads(row["ground_truth"])["question"], "why?")
        self.assertEqual(row["env_config"]["env_configs"][0]["env_name"], "partner_model")

    def test_opening_goes_into_the_prompt_not_the_completion(self):
        items = [{"question": "why?"}]
        rows = data.build_rows(items, system="", user=lambda i: i["question"], opening=lambda i: "Student: huh?")
        self.assertIn("Student: huh?", rows[0]["messages"][-1]["content"])
        self.assertEqual(json.loads(rows[0]["ground_truth"])["opening"], "Student: huh?")

    def test_split_by_source(self):
        items = [{"state": "TX"}, {"state": "PA"}, {"state": "CA"}]
        train, held = data.split_by(items, "state", ["PA"])
        self.assertEqual(len(train), 2)
        self.assertEqual(held, [{"state": "PA"}])

    def test_round_trip_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.jsonl")
            data.write_jsonl([{"a": 1}], path)
            self.assertEqual(data.read_jsonl(path), [{"a": 1}])


class TestAnchor(unittest.TestCase):
    def setUp(self):
        # three items; the "policy" emits the item's keyword, and the outcome is
        # 1.0 only when the text mentions THIS item's keyword
        self.items = [{"key": k} for k in ("alpha", "beta", "gamma")]

    def test_specific_help_scores_gain_and_specificity(self):
        async def policy(items):
            return [i["key"] for i in items]

        async def outcome(item, text):
            return float(item["key"] in text)

        result = run(anchor.Anchor(self.items, policy, outcome).run())
        self.assertEqual(result.baseline, 0.0)
        self.assertEqual(result.treated, 1.0)
        self.assertEqual(result.swapped, 0.0)
        self.assertEqual(result.gain, 1.0)
        self.assertEqual(result.specificity, 1.0)

    def test_generic_help_scores_gain_but_no_specificity(self):
        """The case the anchor exists to catch: it helps, but not because it taught."""

        async def policy(items):
            return ["read the options carefully"] * len(items)

        async def outcome(item, text):
            return 1.0 if text else 0.0

        result = run(anchor.Anchor(self.items, policy, outcome).run())
        self.assertEqual(result.gain, 1.0)
        self.assertEqual(result.specificity, 0.0)

    def test_extra_metric_reports_the_clean_subset(self):
        async def policy(items):
            return ["clean", "flagged", "clean"]

        async def outcome(item, text):
            return 1.0 if text == "flagged" else 0.0

        result = run(
            anchor.Anchor(
                self.items, policy, outcome, extra_metrics={"leaked": lambda i, t: float(t == "flagged")}
            ).run()
        )
        self.assertAlmostEqual(result.extras["leaked"], 1 / 3)
        # restricted to the two unflagged items, the outcome is 0 - the headline
        # 1/3 was entirely the flagged one
        self.assertEqual(result.extras["clean_leaked"], 0.0)

    def test_moved_is_reported_in_standard_errors(self):
        before = anchor.AnchorResult(n=200, baseline=0.5, treated=0.50, swapped=0.5)
        after = anchor.AnchorResult(n=200, baseline=0.5, treated=0.51, swapped=0.5)
        self.assertIn("did not move", anchor.moved(before, after))
        big = anchor.AnchorResult(n=200, baseline=0.5, treated=0.70, swapped=0.5)
        self.assertIn("-> moved", anchor.moved(before, big))


class TestFunctionScorer(unittest.TestCase):
    def test_plain_float(self):
        self.assertEqual(run(FunctionScorer(lambda s: 0.3).score(Sample(completion=""))).score, 0.3)

    def test_dict_return_splits_into_score_and_info(self):
        result = run(FunctionScorer(lambda s: {"score": 0.4, "why": 2.0}).score(Sample(completion="")))
        self.assertEqual(result.score, 0.4)
        self.assertEqual(result.info["why"], 2.0)


if __name__ == "__main__":
    unittest.main()
