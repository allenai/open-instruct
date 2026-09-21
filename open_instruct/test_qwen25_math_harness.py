import pytest

from open_instruct import qwen25_math_harness


def _strip(value: str, skip_unit: bool = False) -> str:
    return value.replace(" ", "") + ("" if skip_unit else "!")


def _extract(text: str, data_name: str) -> str:
    return text.rsplit("boxed{", 1)[-1].rstrip("}") if "boxed{" in text else ""


def _equal(prediction: str, reference: str, timeout: bool = False) -> bool:
    return prediction == reference


FAKE_HARNESS = (_extract, _strip, _equal)


def _responses(pattern: list[list[bool]]) -> list[dict]:
    rows = []
    for p, samples in enumerate(pattern):
        for k, correct in enumerate(samples):
            rows.append(
                {
                    "prompt_index": p,
                    "sample_index": k,
                    "text": "\\boxed{1}" if correct else "\\boxed{2}",
                    "ground_truth": "1",
                    "finish_reason": "length" if (p, k) == (0, 0) else "stop",
                    "response_tokens": 100,
                }
            )
    return rows


class TestGroundTruth:
    def test_minerva_keeps_the_label_apart_from_the_three_aliases(self):
        assert qwen25_math_harness.normalize_ground_truth("x \\neq 2", "minerva_math", _strip) == "x \\ne 2"

    def test_other_sets_go_through_strip_string(self):
        assert qwen25_math_harness.normalize_ground_truth("3 / 4", "math", _strip) == "3/4!"

    def test_every_default_set_maps_to_a_harness_data_name(self):
        for name in qwen25_math_harness.DEFAULT_SETS:
            assert name in qwen25_math_harness.HARNESS_DATA_NAME
            assert name in qwen25_math_harness.PAPER_OPD["qwen3-4b-base"]


class TestGradeAndAggregate:
    def test_inline_grading_and_avg_pass_at_n(self):
        rows = _responses([[True, False, False, True], [False, False, False, False], [True, True, True, True]])
        graded, timeouts = qwen25_math_harness.grade(rows, "math", FAKE_HARNESS, workers=0)
        assert timeouts == 0
        assert [r["prediction"] for r in graded[:2]] == ["1", "2"]
        summary = qwen25_math_harness.aggregate(graded)
        assert summary["problems"] == 3 and summary["samples_per_problem"] == 4
        assert summary["avg_at_n"] == pytest.approx(100 * 6 / 12)
        assert summary["pass_at_n"] == pytest.approx(100 * 2 / 3)
        assert summary["per_sample_index_acc"] == pytest.approx([100 * 2 / 3, 100 / 3, 100 / 3, 100 * 2 / 3])
        assert summary["truncated_fraction"] == pytest.approx(1 / 12)

    def test_uneven_sample_counts_are_rejected(self):
        rows = _responses([[True, True], [True]])
        for r in rows:
            r["correct"] = True
        with pytest.raises(ValueError, match="same number of samples"):
            qwen25_math_harness.aggregate(rows)

    def test_summary_table_includes_the_paper_row_and_means(self):
        results = {
            "math_500": {"problems": 500, "avg_at_n": 78.0, "pass_at_n": 90.0, "truncated_fraction": 0.01},
            "aime24": {"problems": 30, "avg_at_n": 18.0, "pass_at_n": 26.0, "truncated_fraction": 0.1},
        }
        table = qwen25_math_harness.summary_table(results, qwen25_math_harness.PAPER_OPD["qwen3-4b-base"])
        assert "| math_500 | 500 | 78.00 | 90.00 | 78.81 | 90.8 | 1.0% |" in table
        assert "| **mean** | | 48.00 | 58.00 | 48.57 | 58.73 | |" in table
