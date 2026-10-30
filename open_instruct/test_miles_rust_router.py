"""CPU checks for Core's standard-router launch and override contract."""

import dataclasses
from pathlib import Path

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.test_miles_policy_refresh import configured
from open_instruct.test_miles_run_spec import document


@pytest.mark.parametrize("replay", [False, True])
def test_structured_runs_default_to_rust_with_controller_owned_recovery(tmp_path, replay):
    spec = RunSpec.from_dict(
        document(tmp_path) | {"miles": {"use_rollout_routing_replay": replay}}, config_path=tmp_path / "run.toml"
    )
    config = spec.compile()
    options = config.resolved_miles()
    assert not options.get("use_miles_router", False)
    assert options["router_policy"] == "round_robin"
    for name in ("router_disable_retries", "router_disable_health_check", "router_disable_circuit_breaker"):
        assert options[name] is True
        assert "--" + name.replace("_", "-") in config.arguments()
    assert "--use-miles-router" not in config.arguments()


@pytest.mark.parametrize(
    "name", ["router_disable_retries", "router_disable_health_check", "router_disable_circuit_breaker"]
)
def test_explicit_conflicting_recovery_override_is_rejected(name):
    config = RunConfig(CoreConfig(), {name: False})
    with pytest.raises(ValueError, match=name):
        config.resolved_miles()


def test_refresh_accepts_rust_with_replay_and_preserves_python_override():
    config = configured()
    rust = dataclasses.replace(
        config, miles=config.miles | {"use_miles_router": False, "use_rollout_routing_replay": True}
    )
    rust.validate()
    assert "--use-miles-router" not in rust.arguments()
    config.validate()
    assert "--use-miles-router" in config.arguments()
    assert "router_disable_health_check" not in config.resolved_miles()


@pytest.mark.parametrize("key", ["router_policy", "sglang_router_policy"])
def test_explicit_routing_policy_is_preserved(key):
    config = RunConfig(CoreConfig(), {key: "power_of_two"})
    assert config.resolved_miles()[key] == "power_of_two"


def test_three_node_template_keeps_admission_and_selects_rust():
    path = Path(__file__).resolve().parents[1] / "configs/miles/examples/three-node.toml"
    config = RunSpec.load(path).compile()
    options = config.resolved_miles()
    assert options["async_max_concurrent_samples"] == 4864
    assert options["use_miles_router"] is False
    assert options["router_policy"] == "round_robin"
    assert options["router_disable_retries"] is True
