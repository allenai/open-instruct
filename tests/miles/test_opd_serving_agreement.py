"""Reject shifted or incomplete token scoring instead of reporting spurious agreement."""

from types import SimpleNamespace

import pytest
from scripts.miles import compare_opd_complete_repeats, compare_opd_serving_agreement, probe_opd_serving_agreement


def test_response_scores_use_exact_tail_tokens():
    meta = {"input_token_logprobs": [[None, 1], [-0.1, 2], [-0.2, 3], [-0.3, 4]]}
    assert probe_opd_serving_agreement.parse_input_scores(meta, [1, 2, 3, 4], 2) == [-0.2, -0.3]


@pytest.mark.parametrize("entries", [[[-0.2, 2], [-0.3, 3]], [[-0.3, 4]]])
def test_shifted_or_missing_scores_fail(entries):
    with pytest.raises(ValueError, match="align"):
        probe_opd_serving_agreement.parse_input_scores({"input_token_logprobs": entries}, [1, 2, 3, 4], 2)


def test_agreement_metrics_do_not_confuse_signed_bias_with_error():
    result = compare_opd_serving_agreement.difference([-1.0, -1.0], [-1.1, -0.9])
    assert result["abs_mean"] == pytest.approx(0.1)
    assert result["signed_mean_reference_minus_candidate"] == pytest.approx(0)
    assert result["ratio_outside_08_128_fraction"] == 0


def test_nonfinite_or_misaligned_comparisons_fail():
    for candidate in ([float("nan")], [-1, -2]):
        with pytest.raises(ValueError, match="aligned"):
            compare_opd_serving_agreement.difference([-1], candidate)


def test_complete_repeats_separate_answer_changes_from_correctness():
    questions = [{"id": str(i)} for i in range(8)]
    rows = [
        {"id": str(i), "wave": wave, "response": "right", "output_ids": [1], "finish": "stop"}
        for wave, copies in [("repeat-r0", 1), ("repeat-r1", 1), ("reversed", 1), ("single", 1), ("crowded", 2)]
        for _ in range(copies)
        for i in range(8)
    ]
    grades = {(str(i), "right"): True for i in range(8)}
    grades[("0", "wrong")] = False
    rows[8].update(response="wrong", output_ids=[2])
    rows[9].update(output_ids=[3])
    result = compare_opd_complete_repeats.summarize(rows, questions, grades)
    assert result["waves"]["repeat-r1"]["token_changed"] == 2
    assert result["waves"]["repeat-r1"]["right_to_wrong"] == 1
    assert result["waves"]["repeat-r1"]["wrong_to_right"] == 0
    assert result["questions_with_variable_correctness"] == ["0"]
    with pytest.raises(ValueError, match="Missing or duplicated"):
        compare_opd_complete_repeats.summarize(rows[:-1], questions, grades)


def test_vllm_provenance_uses_unwrapped_nested_language_model():
    class LanguageModel:
        lm_head = SimpleNamespace(weight=SimpleNamespace(dtype="bfloat16"))

        def compute_logits(self, hidden):
            return hidden

    model = SimpleNamespace(language_model=LanguageModel())
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=object(), get_model=lambda: model))
    info = probe_opd_serving_agreement.vllm_worker_provenance(worker)
    assert info["head_dtype"] == "bfloat16"
    assert not info["patch_marker"]
    assert not info["language_model_patch_marker"]
    assert info["compute_logits_name"].endswith("LanguageModel.compute_logits")
