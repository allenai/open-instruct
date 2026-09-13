"""Unit tests for preference-quality helpers and shuffle un-mapping."""

import json
import tempfile
import unittest
from pathlib import Path

from open_instruct.preference_quality import (
    filter_preference_jsonl,
    filter_preference_rows,
    is_identical_preference,
    judges_agree,
    score_margin,
    should_swap_after_shuffle,
)


def _chat(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


class TestShouldSwapAfterShuffle(unittest.TestCase):
    def test_truth_table_matches_oracle(self):
        # Oracle: swap iff the judge's preferred display index maps to pair[1].
        cases = [
            ("0", 0, False),
            ("0", 1, True),
            ("1", 0, True),
            ("1", 1, False),
        ]
        for preferred, shuffled_index, expect_swap in cases:
            with self.subTest(preferred=preferred, shuffled_index=shuffled_index):
                self.assertEqual(
                    should_swap_after_shuffle(preferred, shuffled_index),
                    expect_swap,
                )

    def test_strips_whitespace(self):
        self.assertTrue(should_swap_after_shuffle(" 1\n", 0))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            should_swap_after_shuffle("2", 0)
        with self.assertRaises(ValueError):
            should_swap_after_shuffle("0", 2)

    def test_old_guard_never_fired(self):
        """Regression: the previous AND-of-contradictions guard was always False."""
        for preferred in ("0", "1"):
            for shuffled_index in (0, 1):
                old = (
                    preferred == "0"
                    and shuffled_index == 1
                    and preferred == "1"
                    and shuffled_index == 0
                )
                self.assertFalse(old)


class TestPreferenceFilters(unittest.TestCase):
    def test_drop_identical_pairs(self):
        rows = [
            {"chosen": _chat("q", "same"), "rejected": _chat("q", "same")},
            {"chosen": _chat("q", "good"), "rejected": _chat("q", "bad")},
        ]
        kept, stats = filter_preference_rows(rows, drop_identical=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_identical, 1)
        self.assertEqual(kept[0]["chosen"][-1]["content"], "good")

    def test_min_score_margin(self):
        rows = [
            {
                "chosen": _chat("q", "a"),
                "rejected": _chat("q", "b"),
                "chosen_score": 0.9,
                "rejected_score": 0.85,
            },
            {
                "chosen": _chat("q", "c"),
                "rejected": _chat("q", "d"),
                "chosen_score": 0.9,
                "rejected_score": 0.1,
            },
        ]
        kept, stats = filter_preference_rows(rows, min_score_margin=0.2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_low_margin, 1)
        self.assertEqual(kept[0]["chosen"][-1]["content"], "c")

    def test_judge_disagreement(self):
        rows = [
            {
                "chosen": _chat("q", "a"),
                "rejected": _chat("q", "b"),
                "judge_labels": ["0", "1", "0"],
            },
            {
                "chosen": _chat("q", "c"),
                "rejected": _chat("q", "d"),
                "judge_labels": ["1", "1"],
            },
        ]
        kept, stats = filter_preference_rows(
            rows,
            judge_label_key="judge_labels",
            require_judge_agreement=True,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_disagreement, 1)
        self.assertFalse(judges_agree(["0", "1"]))
        self.assertTrue(judges_agree(["1", "1", "1"]))

    def test_length_bias_stats(self):
        rows = [
            {"chosen": _chat("q", "aaaa"), "rejected": _chat("q", "bb")},
            {"chosen": _chat("q", "cccc"), "rejected": _chat("q", "dd")},
        ]
        _, stats = filter_preference_rows(rows)
        self.assertEqual(stats.mean_chosen_len, 4.0)
        self.assertEqual(stats.mean_rejected_len, 2.0)
        self.assertEqual(stats.length_bias_ratio, 2.0)

    def test_score_margin_helper(self):
        self.assertEqual(score_margin({"a": 1.0, "b": 0.25}, "a", "b"), 0.75)
        self.assertIsNone(score_margin({}, "a", "b"))

    def test_is_identical_with_string_arms(self):
        self.assertTrue(is_identical_preference({"chosen": "x", "rejected": "x"}))
        self.assertFalse(is_identical_preference({"chosen": "x", "rejected": "y"}))

    def test_jsonl_roundtrip(self):
        rows = [
            {"chosen": _chat("q", "same"), "rejected": _chat("q", "same")},
            {"chosen": _chat("q", "good"), "rejected": _chat("q", "bad")},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.jsonl"
            out = tmp_path / "out.jsonl"
            stats_path = tmp_path / "stats.json"
            with inp.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            stats = filter_preference_jsonl(inp, out, stats_path=stats_path)
            kept = [json.loads(line) for line in out.read_text().splitlines() if line]
            self.assertEqual(len(kept), 1)
            self.assertEqual(stats.kept_rows, 1)
            payload = json.loads(stats_path.read_text())
            self.assertEqual(payload["dropped_identical"], 1)


if __name__ == "__main__":
    unittest.main()
