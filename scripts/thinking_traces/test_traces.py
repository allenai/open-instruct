"""Tests for thinking-trace parsing and the length statistics built on top of it."""

import numpy as np
import pytest
from scripts.thinking_traces import analyze_traces, generate_traces

# The two chat templates in the experiment disagree about the opening tag:
# Qwen3 lets the model emit <think> itself, while DeepSeek-R1-Distill prefills
# it in the assistant prefix. Both shapes have to parse to the same thing.
QWEN_STYLE = "<think>\nlet me work through this.\n</think>\n\nThe answer is 4."
DEEPSEEK_STYLE = "\nlet me work through this.\n</think>\n\nThe answer is 4."


@pytest.mark.parametrize("text", [QWEN_STYLE, DEEPSEEK_STYLE])
def test_split_trace_handles_both_templates(text):
    thinking, answer, kind = generate_traces.split_trace(text, "stop")
    assert kind == generate_traces.KIND_CLOSED
    assert thinking.strip() == "let me work through this."
    assert answer.strip() == "The answer is 4."


def test_split_trace_truncated_is_censored_not_empty():
    """A trace cut off at the token cap is still a trace, just a lower bound."""
    thinking, answer, kind = generate_traces.split_trace("<think>\nstill thinking and thi", "length")
    assert kind == generate_traces.KIND_TRUNCATED
    assert thinking.strip() == "still thinking and thi"
    assert answer == ""


def test_split_trace_truncated_without_opening_tag():
    thinking, answer, kind = generate_traces.split_trace("still thinking and thi", "length")
    assert kind == generate_traces.KIND_TRUNCATED
    assert thinking.strip() == "still thinking and thi"
    assert answer == ""


def test_split_trace_no_thinking_block_counts_as_zero():
    """A model that answers outright has a zero-length trace, not a truncated one."""
    thinking, answer, kind = generate_traces.split_trace("The answer is 4.", "stop")
    assert kind == generate_traces.KIND_NO_BLOCK
    assert thinking == ""
    assert answer == "The answer is 4."


def test_split_trace_ignores_think_tag_inside_the_answer():
    text = "<think>\nreasoning\n</think>\n\nYou write it as <think> in the prompt."
    thinking, answer, kind = generate_traces.split_trace(text, "stop")
    assert kind == generate_traces.KIND_CLOSED
    assert thinking.strip() == "reasoning"
    assert "<think>" in answer


def _records(groups):
    return [
        {"prompt_index": i, "prompt_sha": f"sha{i}", "thinking_tokens": v, "dataset_source": "s", "answer_tokens": 1}
        for i, values in enumerate(groups)
        for v in values
    ]


def test_variance_decomposition_pure_between_prompt():
    """Identical samples within each prompt: all variance is between prompts."""
    result = analyze_traces.decompose_variance(_records([[10, 10], [20, 20], [30, 30]]))
    assert result["within_prompt_var"] == pytest.approx(0.0)
    assert result["between_prompt_var"] == pytest.approx(100.0)
    assert result["intraclass_correlation"] == pytest.approx(1.0)


def test_variance_decomposition_pure_within_prompt():
    """Identical prompt means: no between-prompt signal, ICC pinned at zero."""
    result = analyze_traces.decompose_variance(_records([[10, 20], [10, 20], [10, 20]]))
    assert result["within_prompt_var"] == pytest.approx(50.0)
    assert result["between_prompt_var"] == pytest.approx(0.0)
    assert result["intraclass_correlation"] == pytest.approx(0.0)


def test_summarize_reports_truncation_and_moments():
    records = _records([[100, 200], [300, 400]])
    for record in records:
        record.update({"model": "m", "kind": "closed", "truncated": False, "finish_reason": "stop"})
    records[0]["truncated"] = True
    records[0]["kind"] = "truncated"

    args = analyze_traces.argparse.Namespace(min_per_source=1, seed=0)
    summary = analyze_traces.summarize("m", records, args)

    lengths = np.array([100, 200, 300, 400], dtype=float)
    assert summary["n_traces"] == 4
    assert summary["n_prompts"] == 2
    assert summary["mean"] == pytest.approx(lengths.mean())
    assert summary["variance"] == pytest.approx(lengths.var(ddof=1))
    assert summary["truncation_rate"] == pytest.approx(0.25)
    # The censored trace is excluded from the completed-only figures.
    assert summary["completed_only"]["n"] == 3
    assert summary["completed_only"]["mean"] == pytest.approx(300.0)
    assert summary["mean_ci95"][0] <= summary["mean"] <= summary["mean_ci95"][1]


def test_analyze_kind_constant_stays_in_sync_with_generator():
    """analyze_traces duplicates this constant to stay numpy-only; keep them equal."""
    assert analyze_traces.KIND_CLOSED == generate_traces.KIND_CLOSED


def test_summarize_separates_answer_truncation_from_trace_truncation():
    """Hitting the cap after </think> leaves the trace length exact."""
    records = _records([[100, 200], [300, 400]])
    for record in records:
        record.update({"model": "m", "kind": "closed", "truncated": False, "finish_reason": "stop"})
    # Closed trace, but the completion ran out of budget while writing the answer.
    records[1].update({"finish_reason": "length"})
    # Genuinely censored trace: no closing tag before the cap.
    records[2].update({"finish_reason": "length", "kind": "truncated", "truncated": True})

    args = analyze_traces.argparse.Namespace(min_per_source=1, seed=0)
    summary = analyze_traces.summarize("m", records, args)

    assert summary["n_answer_truncated_after_complete_trace"] == 1
    assert summary["n_truncated"] == 1
    assert summary["completed_only"]["n"] == 3


def test_compare_detects_a_real_shift_and_verifies_prompt_identity():
    base = _records([[100, 110], [200, 210], [300, 310], [400, 410]])
    longer = [dict(r, thinking_tokens=r["thinking_tokens"] * 2) for r in base]

    result = analyze_traces.compare("a", base, "b", longer, seed=0)

    assert result["prompt_sets_identical"] is True
    assert result["n_shared_prompts"] == 4
    assert result["mean_difference_b_minus_a"] == pytest.approx(result["mean_a"])
    assert result["ratio_b_over_a"] == pytest.approx(2.0)
    assert result["share_of_prompts_where_b_longer"] == pytest.approx(1.0)


def test_compare_flags_mismatched_prompt_sets():
    a = _records([[100, 110], [200, 210]])
    b = _records([[100, 110], [200, 210]])
    for record in b:
        record["prompt_sha"] = record["prompt_sha"] + "-different"
    result = analyze_traces.compare("a", a, "b", b, seed=0)
    assert result["prompt_sets_identical"] is False
    assert result["n_shared_prompts"] == 0
