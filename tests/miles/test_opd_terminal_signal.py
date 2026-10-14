import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / "scripts/miles/analyze_opd_terminal_signal.py"
_SPEC = importlib.util.spec_from_file_location("terminal_signal", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def test_tied_ranks_and_repeated_windows():
    assert analysis.spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1)
    assert analysis.spearman([1, 1], [2, 3]) is None
    assert analysis.spearman([1, None], [2, 3]) is None
    assert analysis.repeated_window_fraction([1] * 64) == pytest.approx(32 / 33)
    assert analysis.repeated_window_fraction(list(range(64))) == 0
    assert analysis.repeated_window_fraction([1, 2]) is None


def test_masked_eos_is_excluded_and_length_bins_count_responses():
    advantages = [torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0])]
    data = {
        "tokens": [[10, 20, 99], [10, 30, 99]],
        "response_lengths": [2, 2],
        "advantages": advantages,
        "loss_masks": [[1, 1], [1, 0]],
        "teacher_log_probs": [value.clone() for value in advantages],
        "rollout_log_probs": [torch.zeros(2), torch.zeros(2)],
        "truncated": [False, False],
    }
    result = analysis.summarize(data, [99])
    assert result["all_active_tokens"]["count"] == 3
    assert result["active_eos_tokens"]["count"] == 1
    assert result["active_eos_tokens"]["mean"] == -2
    assert result["active_non_eos_tokens"]["mean"] == -1
    assert result["length_bins_lower_inclusive_upper_exclusive"]["0"]["responses"] == 2
    data["advantages"][0] = torch.zeros(2)
    with pytest.raises(ValueError, match="pure teacher-minus-rollout"):
        analysis.summarize(data, [99])
