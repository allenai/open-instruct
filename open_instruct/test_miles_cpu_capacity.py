"""Ray CPU scheduling capacity must be valid before allocation or service startup."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import cluster

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/miles/examples/three-node.toml"


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "", "true", " 64", "nan"])
def test_bad_ray_cpu_setting_fails_planning_and_runtime_before_startup(monkeypatch, value):
    document = RunSpec.load(EXAMPLE).to_dict()
    document["launch"]["env"]["OI_MILES_RAY_CPUS"] = value
    with pytest.raises(InputError, match="OI_MILES_RAY_CPUS"):
        RunSpec.from_dict(document)
    spec = Mock()
    monkeypatch.setattr(cluster.RunSpec, "load", lambda path: spec)
    monkeypatch.setenv("OI_MILES_RAY_CPUS", value)
    with pytest.raises(InputError, match="OI_MILES_RAY_CPUS"):
        cluster.run("unused.toml")
    spec.compile.assert_not_called()
