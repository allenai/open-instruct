"""Guard unchanged weights, response alignment, and probability-ratio direction."""

import pytest
import torch
from scripts.miles import probe_lm_head_precision as probe


def test_pinned_core_factory_entry_points():
    assert callable(probe.olmo3.build_olmo3_moe_config_from_hf_config)
    assert callable(probe.olmo3.load_olmo3_moe_hf_state)


def test_strict_projection_preserves_weights_and_fp32_output():
    weight = torch.tensor([[1.0, 0.5], [0.25, -0.5], [1.0, -1.0]], dtype=torch.bfloat16)
    hidden = torch.tensor([[[0.3, 0.7], [0.1, 0.9]]], dtype=torch.bfloat16)
    original = weight.clone()
    output = probe.head_logits(hidden, weight, strict=True)
    assert output.shape == (1, 2, 3)
    assert output.dtype == torch.float32
    assert torch.equal(weight, original)
    assert weight.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.nn.functional.linear(hidden.float(), weight.float()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP32-output BF16 GEMM requires CUDA")
def test_cuda_fp32_output_matches_exact_small_projection():
    weight = torch.tensor([[1.0, 0.5], [0.25, -0.5]], dtype=torch.bfloat16, device="cuda")
    hidden = torch.tensor([[[0.3, 0.7], [0.1, 0.9]]], dtype=torch.bfloat16, device="cuda")
    before = weight.clone()
    actual = probe.head_logits(hidden, weight)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, probe.head_logits(hidden, weight, strict=True), rtol=0, atol=0)
    assert torch.equal(weight, before)


def test_selected_scores_skip_prompt_and_final_prediction():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [4.0, 0.0, 0.0]]])
    actual = probe.selected_scores(logits, [2, 1, 0, 2], start=1)
    expected = logits.float().log_softmax(-1)[0, [1, 2], [0, 2]]
    torch.testing.assert_close(actual, expected)


def test_ratio_tail_distinguishes_tis_from_twenty_percent_band():
    serving = torch.tensor([0.1, 0.2, 0.5]).log()
    core = torch.tensor([0.3, 0.26, 0.25]).log()
    result = probe.probability_statistics(serving, core)
    assert result["tis_upper2_fraction"] == pytest.approx(1 / 3)
    assert result["ratio_outside_20pct_fraction"] == 1
    assert result["ratio_min"] == pytest.approx(0.5)
    assert result["ratio_max"] == pytest.approx(3)
