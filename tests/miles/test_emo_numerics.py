"""Keep diagnostic comparisons aligned to actual response token positions."""

import math

import pytest
import torch
from scripts.miles import emo_numerics


def test_response_score_shift():
    logits = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [9.0, 0.0, 0.0]]])
    tokens = torch.tensor([[0, 0, 1, 2]])
    actual = emo_numerics.selected_scores(logits, tokens, prompt_length=2)
    expected = [-math.log1p(2 * math.exp(-2)), -math.log1p(2 * math.exp(-3))]
    assert actual == pytest.approx([float(x) for x in expected])


def test_comparison_reports_signed_bias_and_rejects_misalignment():
    result = emo_numerics.difference([-1.0, -1.0], [-0.5, -1.5])
    assert result == {"mean_abs": 0.5, "max_abs": 0.5, "signed_mean": 0.0}
    with pytest.raises(ValueError, match="aligned finite"):
        emo_numerics.difference([1.0, 2.0], [1.0])
