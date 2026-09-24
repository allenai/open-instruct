"""Rollout capture is opt-in and operational, with whole-group rates."""

import pytest

from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec

EXAMPLE = "configs/miles/examples/small.toml"


def test_capture_defaults_off():
    config = RunSpec.load(EXAMPLE).compile()
    assert config.miles["rollout_sample_rate"] == 0


@pytest.mark.parametrize("rate, expected", [(0, 0), (0.1, 0.1), (1, 1), (9, 1)])
def test_capture_rate_reaches_native_runtime(rate, expected):
    spec = RunSpec.load(EXAMPLE, overrides=[f"output.rollout_sample_rate={rate}"])
    config = spec.compile()
    assert config.miles["rollout_sample_rate"] == expected
    assert config.miles["save_debug_rollout_data"] == f"{spec.output['root']}/rollouts/{{rollout_id}}.pt"
    assert "--rollout-sample-rate" in config.arguments()


@pytest.mark.parametrize("rate", ["-0.1", "nan", "inf", "true", '"0.5"'])
def test_invalid_rate_rejected(rate):
    with pytest.raises(InputError, match="output.rollout_sample_rate"):
        RunSpec.load(EXAMPLE, overrides=[f"output.rollout_sample_rate={rate}"])


@pytest.mark.parametrize("section", ["training", "miles"])
def test_old_capture_path_setting_has_migration_message(section):
    with pytest.raises(InputError, match="use output.rollout_sample_rate"):
        RunSpec.load(EXAMPLE, overrides=[f'{section}.save_debug_rollout_data="/tmp/old/{{rollout_id}}.pt"']).compile()
