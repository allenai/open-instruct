"""Check token alignment, ties, and probability interpretation in the probe."""

import pytest
import torch
from scripts.miles import probe_core_choices as probe


def test_core_choice_rank_and_margin():
    result = probe.core_record(torch.tensor([[2.0, 2.0, 1.0], [0.0, 1.0, 3.0]]), [1, 0])
    assert result["argmax_ids"] == [0, 2]
    assert result["chosen_rank"] == [1, 3]
    assert result["chosen_margin"] == [0.0, 3.0]
    assert result["top2_margin"] == [0.0, 2.0]
    torch.testing.assert_close(
        torch.tensor(result["logprobs"]),
        torch.tensor([[2.0, 2.0, 1.0], [0.0, 1.0, 3.0]]).log_softmax(-1)[[0, 1], [1, 0]],
    )


def test_forced_scores_align_after_prompt_and_reject_shift():
    result = {
        "meta_info": {
            "input_token_logprobs": [[None, 10], [-0.1, 11], [-0.4, 12]],
            "input_top_logprobs": [None, [[-0.1, 11], [-2.0, 10]], [[-0.2, 13], [-0.4, 12]]],
        }
    }
    record = probe.serving_record(result, [10, 11, 12], 2, forced=True)
    assert record["output_ids"] == [12]
    assert record["argmax_ids"] == [13]
    assert record["logprobs"] == [-0.4]
    with pytest.raises(ValueError, match="alignment"):
        probe.serving_record(result, [11, 12, 13], 2, forced=True)


def test_equal_chosen_probabilities_do_not_imply_same_argmax():
    actual = {"logprobs": [-1.0, -2.0], "argmax_ids": [10, 20]}
    core = {"logprobs": [-1.0, -2.0], "argmax_ids": [10, 30]}
    report = probe.comparison(actual, core)
    assert report["argmax_disagreements"] == [1]
    assert report["logprobs"]["max_abs"] == 0


def test_cached_greedy_uses_actual_token_when_topk_order_breaks_ties_differently():
    result = {"meta_info": {"output_token_logprobs": [[-0.7, 10]], "output_top_logprobs": [[[-0.7, 20], [-0.7, 10]]]}}
    record = probe.serving_record(result, [10], 2, forced=False)
    assert record["argmax_ids"] == [10]
    assert record["top2_tied"] == [True]
