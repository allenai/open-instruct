"""Matched experiment controls and explicit opt-in compilation switches."""

import dataclasses
from pathlib import Path

import pytest
from scripts.miles import launch_trainer_capacity, trainer_capacity_config

from open_instruct.miles.configuration.config import CoreConfig


@pytest.mark.parametrize("variant", trainer_capacity_config.VARIANTS)
def test_variants_preserve_inputs_and_dynamic_row_safeguard(variant, tmp_path):
    run = trainer_capacity_config.configuration(variant, str(tmp_path))
    baseline = trainer_capacity_config.configuration("baseline", str(tmp_path))
    assert run.miles == baseline.miles
    assert run.core.row_specialization == "dynamic"
    assert run.core.sequence_packing
    assert run.core.max_sequence_length == 6144
    assert run.miles["global_batch_size"] == 128
    assert run.miles["num_rollout"] == 16
    assert run.miles["hf_checkpoint"].endswith("20260910-core-megatron-v1/hf")
    assert run.miles["use_rollout_routing_replay"]
    expected = (
        {}
        if variant == "baseline"
        else trainer_capacity_config.VARIANTS["lean"] | trainer_capacity_config.VARIANTS[variant]
    )
    assert dataclasses.asdict(run.core) == dataclasses.asdict(dataclasses.replace(baseline.core, **expected))
    assert run.arguments()


@pytest.mark.parametrize("field", ["compile_model", "compile_optimizer", "use_reduce_scatter"])
def test_optimization_switches_are_explicit_booleans(field):
    assert getattr(CoreConfig(), field) is False
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: "false"})


def test_launch_is_bounded_ep2_and_preserves_reports():
    spec = launch_trainer_capacity.specification("image", "source", "digest", "baseline", Path("/weka/test"), "commit")
    task = spec["tasks"][0]
    assert task["resources"]["gpuCount"] == 2
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"] == dict(priority="urgent", minRuntime="1h", autoResume=False)
    assert task["timeout"] == "2h"
    assert "trap" in task["arguments"][0]
    assert "trainer_capacity_rank.sh" in task["arguments"][0]


@pytest.mark.parametrize("variant", trainer_capacity_config.VARIANTS)
def test_kernel_switches_are_isolated(variant):
    environment = trainer_capacity_config.environment(variant)
    assert sum(value == "1" for value in environment.values()) == int(
        variant in {"vector-grad-add", "pairwise-swiglu"}
    )
    if variant == "pairwise-swiglu":
        assert environment["OLMO_PROFILE_SWIGLU_PAIRWISE"] == "1"
    if variant == "vector-grad-add":
        assert environment["OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE"] == "1"
