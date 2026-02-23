import pytest

from open_instruct.grpo_utils import ExperimentConfig


def test_local_eval_every_accepts_minus_one():
    cfg = ExperimentConfig(local_eval_every=-1)
    assert cfg.local_eval_every == -1


@pytest.mark.parametrize("value", [0, -2])
def test_local_eval_every_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="local_eval_every"):
        ExperimentConfig(local_eval_every=value)


@pytest.mark.parametrize("value", [0, -1, 3, 6])
def test_eval_pass_at_k_rejects_non_power_of_two_values(value):
    with pytest.raises(ValueError, match="eval_pass_at_k"):
        ExperimentConfig(eval_pass_at_k=value)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_eval_pass_at_k_accepts_power_of_two_values(value):
    cfg = ExperimentConfig(eval_pass_at_k=value)
    assert cfg.eval_pass_at_k == value
