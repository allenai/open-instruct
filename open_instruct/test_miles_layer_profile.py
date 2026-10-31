"""Layer profiling is an explicit, validated run-wide measurement interval."""

import pytest

from open_instruct.miles.configuration.config import CoreConfig


@pytest.mark.parametrize("interval", [-1, True, 1.5, "4"])
def test_layer_profile_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="layer_profile_interval"):
        CoreConfig(layer_profile_interval=interval)


def test_layer_profile_is_opt_in():
    assert CoreConfig().layer_profile_interval == 0
    assert CoreConfig(layer_profile_interval=4).layer_profile_interval == 4
