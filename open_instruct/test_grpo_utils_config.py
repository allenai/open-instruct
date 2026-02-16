import pytest

from open_instruct.grpo_utils import ExperimentConfig


def test_local_eval_every_accepts_minus_one():
    cfg = ExperimentConfig(local_eval_every=-1)
    assert cfg.local_eval_every == -1


@pytest.mark.parametrize("value", [0, -2])
def test_local_eval_every_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="local_eval_every"):
        ExperimentConfig(local_eval_every=value)
