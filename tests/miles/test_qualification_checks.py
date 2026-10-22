"""The resume gate must reject lost optimizer state and nonfinite values."""

from importlib import util
from pathlib import Path

import pytest
import torch

SPEC = util.spec_from_file_location("qualification_runtime", Path(__file__).parent / "qualification/runtime.py")
qualification = util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.mark.parametrize("scale", [0.01, 1e-9])
def test_resume_comparison_rejects_lost_adam_moment(scale):
    expected = {"exp_avg": torch.tensor([scale, 2 * scale]), "step": 2}
    reset = {"exp_avg": torch.zeros(2), "step": 2}
    with pytest.raises(AssertionError):
        qualification.compare_value(expected, reset, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("both_sides", [False, True])
def test_resume_comparison_rejects_nonfinite(value, both_sides):
    with pytest.raises(AssertionError):
        qualification.compare_value(
            torch.tensor([value if both_sides else 0.0]), torch.tensor([value]), rtol=1e-5, atol=1e-6
        )


def test_resume_comparison_rejects_missing_state():
    with pytest.raises(AssertionError):
        qualification.compare_value({"step": 3, "cursor": 3}, {"step": 3}, rtol=0, atol=0)


def test_resume_comparison_rejects_cursor_reset():
    with pytest.raises(AssertionError):
        qualification.compare_value({"cursor": 3}, {"cursor": 0}, rtol=0, atol=0)
