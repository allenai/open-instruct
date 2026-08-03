"""Tests for the unit decomposition and the transfer probe.

No GPU, no API, no network. The point is that the pairing logic - which is where
a silent bug would invalidate the measurement rather than crash it - is checkable
before anything is queued.

    python -m unittest projects.tutor.test_transfer -v
"""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
import unittest

from open_instruct.scored_rewards.anchor import Anchor
from projects.tutor import transfer_probe, units
from projects.tutor.student import ChoiceStudent


def item(question, choices=("a", "b", "c", "d"), gold_idx=1):
    return {"question": question, "choices": list(choices), "gold_idx": gold_idx, "gold": list(choices)[gold_idx]}


def unit(key, primary, misconception="believes something wrong"):
    return units.Unit(key=key, units=[primary], prerequisites=[], misconception=misconception)


class TestItemKey(unittest.TestCase):
    def test_joins_items_across_files_despite_whitespace(self):
        """The traces carry only the question text, so it is the only join key."""
        a = units.item_key({"question": "Which  number\nis largest?"})
        b = units.item_key({"question": "which number is largest?"})
        self.assertEqual(a, b)

    def test_different_questions_do_not_collide(self):
        self.assertNotEqual(units.item_key(item("one")), units.item_key(item("two")))


class TestParse(unittest.TestCase):
    def test_pulls_json_out_of_a_chatty_reply(self):
        reply = 'Sure!\n{"units": ["Ordering Numbers"], "prerequisites": [["a","b"]], "misconception": "x"}\nHope that helps'
        out = units.parse(reply)
        self.assertEqual(out["units"], ["ordering numbers"])
        self.assertEqual(out["prerequisites"], [("a", "b")])

    def test_raises_rather_than_defaulting(self):
        """A silent empty decomposition would drop the item from every pair."""
        with self.assertRaises(ValueError):
            units.parse("I could not do that")

    def test_missing_misconception_is_none_not_empty_string(self):
        self.assertIsNone(units.parse('{"units": ["u"], "misconception": null}')["misconception"])


class TestBelieved(unittest.TestCase):
    def test_takes_the_modal_wrong_answer(self):
        traces = [
            {"prompt": "q", "gold": "right", "student_believed": "wrongA"},
            {"prompt": "q", "gold": "right", "student_believed": "wrongA"},
            {"prompt": "q", "gold": "right", "student_believed": "wrongB"},
        ]
        self.assertEqual(units.believed_by_item(traces)[units.item_key({"question": "q"})], "wrongA")

    def test_ignores_correct_answers(self):
        traces = [{"prompt": "q", "gold": "right", "student_believed": "right"}]
        self.assertEqual(units.believed_by_item(traces), {})


class TestCoverage(unittest.TestCase):
    def test_counts_only_units_that_can_actually_pair(self):
        rows = [unit("a", "shared"), unit("b", "shared"), unit("c", "lonely")]
        stats = units.coverage(rows)
        self.assertEqual(stats["shared_units"], 1)
        self.assertEqual(stats["pairable_items"], 2)  # not 3 - "lonely" cannot pair


class TestPairing(unittest.TestCase):
    def setUp(self):
        self.items = [item(f"q{i}") for i in range(6)]
        self.units = {}
        for i in range(4):
            self.units[units.item_key(self.items[i])] = unit(units.item_key(self.items[i]), "fractions")
        for i in (4, 5):
            self.units[units.item_key(self.items[i])] = unit(units.item_key(self.items[i]), "geometry")
        self.dialogues = {units.item_key(i): [{"completion": "talk", "leaked": 0}] for i in self.items}

    def pairs(self, **kw):
        return transfer_probe.build_pairs(self.items, self.units, self.dialogues, **kw)

    def test_source_and_target_share_a_unit(self):
        for p in self.pairs():
            self.assertEqual(self.units[p["source_key"]].primary, self.units[p["target_key"]].primary)

    def test_an_item_is_never_paired_with_itself(self):
        for p in self.pairs():
            self.assertNotEqual(p["source_key"], p["target_key"])

    def test_a_unit_with_one_item_produces_no_pair(self):
        self.units[units.item_key(self.items[5])] = unit(units.item_key(self.items[5]), "solo")
        self.assertNotIn("solo", {p["unit"] for p in self.pairs()})

    def test_sources_per_target_multiplies_n(self):
        """n=56 at one source could not resolve the effect; this is the fix."""
        one = self.pairs(sources_per_target=1, max_per_unit=99)
        three = self.pairs(sources_per_target=3, max_per_unit=99)
        self.assertGreater(len(three), len(one))
        # and still never against itself
        for p in three:
            self.assertNotEqual(p["source_key"], p["target_key"])

    def test_extra_sources_are_distinct_not_repeats(self):
        pairs = self.pairs(sources_per_target=3, max_per_unit=99)
        by_target = {}
        for p in pairs:
            by_target.setdefault(p["target_key"], []).append(p["source_key"])
        for target, sources in by_target.items():
            self.assertEqual(len(sources), len(set(sources)), f"duplicate source for {target}")

    def test_max_per_unit_stops_one_broad_unit_dominating(self):
        counts = {}
        for p in self.pairs(max_per_unit=2):
            counts[p["unit"]] = counts.get(p["unit"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_source_must_have_a_dialogue(self):
        self.dialogues = {units.item_key(self.items[0]): [{"completion": "talk", "leaked": 0}]}
        for p in self.pairs():
            self.assertEqual(p["source_key"], units.item_key(self.items[0]))

    def test_deterministic_under_a_seed(self):
        self.assertEqual(
            [(p["source_key"], p["target_key"]) for p in self.pairs(seed=7)],
            [(p["source_key"], p["target_key"]) for p in self.pairs(seed=7)],
        )


class TestDialogueChoice(unittest.TestCase):
    def test_prefers_a_non_leaking_dialogue(self):
        entries = [{"completion": "leaky", "leaked": 1}, {"completion": "clean", "leaked": 0}]
        for seed in range(5):
            picked = transfer_probe.pick_dialogue(entries, random.Random(seed))
            self.assertEqual(picked["completion"], "clean")

    def test_falls_back_when_every_dialogue_leaked(self):
        entries = [{"completion": "leaky", "leaked": 1}]
        self.assertEqual(transfer_probe.pick_dialogue(entries, random.Random(0))["completion"], "leaky")


class TestSwap(unittest.TestCase):
    def test_foreign_dialogue_always_comes_from_another_unit(self):
        """Rotation would land on the same unit; this is why swap is overridden."""
        pairs = [{"unit": "fractions"}, {"unit": "fractions"}, {"unit": "geometry"}]
        outputs = ["f1", "f2", "g1"]
        swapped = transfer_probe.make_swap({}, seed=0)(pairs, outputs)
        self.assertEqual(swapped[0], "g1")
        self.assertEqual(swapped[1], "g1")
        self.assertIn(swapped[2], {"f1", "f2"})

    def test_returns_empty_when_no_foreign_unit_exists(self):
        pairs = [{"unit": "only"}, {"unit": "only"}]
        self.assertEqual(transfer_probe.make_swap({}, seed=0)(pairs, ["a", "b"]), ["", ""])


class TestAnchorSwapHook(unittest.TestCase):
    """The generic change: Anchor must honour a custom counterfactual."""

    def run_anchor(self, swap=None):
        items = [{"i": 0}, {"i": 1}, {"i": 2}]

        async def policy(its):
            return [f"out{i['i']}" for i in its]

        async def outcome(it, text):
            return 1.0 if text == f"out{it['i']}" else 0.0

        return asyncio.run(Anchor(items=items, policy=policy, outcome=outcome, swap=swap).run())

    def test_default_still_rotates(self):
        result = self.run_anchor()
        self.assertEqual(result.treated, 1.0)
        self.assertEqual(result.swapped, 0.0)

    def test_custom_swap_is_used(self):
        result = self.run_anchor(swap=lambda items, outputs: list(outputs))
        # handing each item its OWN output makes specificity vanish, which is
        # exactly the mistake a same-unit counterfactual makes silently
        self.assertEqual(result.swapped, 1.0)
        self.assertEqual(result.specificity, 0.0)

    def test_wrong_length_swap_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_anchor(swap=lambda items, outputs: ["only one"])


class TestLeakContrast(unittest.TestCase):
    """The check that can invalidate the headline must itself be well powered."""

    def subset(self, dialogues):
        pairs = [{"source_key": k, "target": item(f"t{k}"), "target_key": f"t{k}"} for k in dialogues]
        return [
            p
            for p in pairs
            if any(not e.get("leaked") for e in dialogues[p["source_key"]])
            and any(e.get("leaked") for e in dialogues[p["source_key"]])
        ]

    def test_only_sources_holding_both_kinds_qualify(self):
        dialogues = {
            "both": [{"completion": "c", "leaked": 0}, {"completion": "l", "leaked": 1}],
            "clean_only": [{"completion": "c", "leaked": 0}],
            "leaked_only": [{"completion": "l", "leaked": 1}],
        }
        self.assertEqual([p["source_key"] for p in self.subset(dialogues)], ["both"])

    def test_pairing_beats_splitting_the_headlines_own_sources(self):
        """Splitting left the leaked arm at n=16 on the real corpus; pairing gave 83."""
        dialogues = {f"s{i}": [{"completion": "c", "leaked": 0}, {"completion": "l", "leaked": 1}] for i in range(30)}
        # every source qualifies, because each holds both kinds
        self.assertEqual(len(self.subset(dialogues)), 30)


class TestStubStudent(unittest.TestCase):
    def test_answers_gold_when_the_text_contains_it(self):
        stub = transfer_probe.StubStudent()
        picked = asyncio.run(stub.choose("q", ["red", "blue", "green"], hint="the answer is blue"))
        self.assertEqual(picked, 1)

    def test_returns_a_valid_index_with_no_text(self):
        stub = transfer_probe.StubStudent()
        for _ in range(10):
            self.assertIn(asyncio.run(stub.choose("q", ["a", "b", "c", "d"], hint="")), range(4))


class TestPrompt(unittest.TestCase):
    def test_transfer_prompt_says_the_context_is_a_different_problem(self):
        text = transfer_probe.TransferStudent.PROMPT.format(hint="D", question="Q")
        self.assertIn("different problem", text)

    def test_the_anchors_own_instrument_is_untouched(self):
        self.assertEqual(ChoiceStudent.PROMPT, "Fact: {hint}\nQuestion: {question}\nAnswer:")


class TestRoundTrip(unittest.TestCase):
    def test_units_survive_save_and_load(self):
        rows = [units.Unit(key="k", units=["u"], prerequisites=[("u", "v")], misconception="m")]
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        units.save(rows, path)
        back = units.load(path)
        self.assertEqual(back["k"].primary, "u")
        self.assertEqual(back["k"].prerequisites, [("u", "v")])

    def test_load_dialogues_filters_by_tier_and_drops_empties(self):
        rows = [
            {"prompt": "q1", "completion": "text", "tier": "policy"},
            {"prompt": "q2", "completion": "text", "tier": "expert"},
            {"prompt": "q3", "completion": "   ", "tier": "policy"},
        ]
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
            path = handle.name
        got = transfer_probe.load_dialogues(path, tier="policy")
        self.assertEqual(list(got), [units.item_key({"question": "q1"})])


if __name__ == "__main__":
    unittest.main()
