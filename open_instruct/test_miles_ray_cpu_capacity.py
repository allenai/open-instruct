"""Ray resource budgets must fit the allocation visible inside each replica."""

import pytest

from open_instruct.miles.execution import cluster


@pytest.mark.parametrize("count", ["0", "-1", "245", "1.5", "bad"])
def test_invalid_cpu_budget(count):
    with pytest.raises(ValueError, match="OI_MILES_RAY_CPUS"):
        cluster.ray_cpu_count({"OI_MILES_RAY_CPUS": count, "BEAKER_ASSIGNED_CPU_COUNT": "244"})


def test_explicit_and_default_cpu_budget():
    assert cluster.ray_cpu_count({"OI_MILES_RAY_CPUS": "64", "BEAKER_ASSIGNED_CPU_COUNT": "244"}) == 64
    assert cluster.ray_cpu_count({"BEAKER_ASSIGNED_CPU_COUNT": "244"}) == 32


def test_default_cpu_budget_fits_small_allocation():
    assert cluster.ray_cpu_count({"BEAKER_ASSIGNED_CPU_COUNT": "16"}) == 16
