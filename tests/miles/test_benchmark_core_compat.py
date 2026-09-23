"""Guard workload identity selection and direction of Core/serving ratios."""

import pytest
import torch
from scripts.miles import benchmark_core_compat as benchmark


def test_selection_interleaves_real_domains_and_deduplicates():
    rows = [
        {"input": f"{verifier}-{index}", "metadata": {"verifiers": [{"name": verifier}]}}
        for verifier in ["math", "code_stdio", "ifeval", "general-quality_ref"]
        for index in range(2)
    ]
    selected = benchmark.select_rows([rows[0], *rows], 2)
    assert [domain for domain, _ in selected] == list(benchmark.DOMAINS) * 2
    assert len({row["input"] for _, row in selected}) == 8
    with pytest.raises(ValueError, match="coverage"):
        benchmark.select_rows(rows, 3)


def test_ratio_is_core_probability_over_serving_probability():
    serving = torch.tensor([0.2, 0.4]).log()
    core = torch.tensor([0.4, 0.2]).log()
    result = benchmark.summarize_delta(serving, core)
    assert result["ratio_min"] == pytest.approx(0.5)
    assert result["ratio_max"] == pytest.approx(2)
    assert result["ratio_outside_20pct_fraction"] == 1
    assert result["mean_serving_minus_core"] == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        benchmark.summarize_delta(torch.tensor([float("nan")]), torch.zeros(1))
