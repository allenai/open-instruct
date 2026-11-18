"""Offline boundaries for Core Ulysses, independent of inference parallelism."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import constraints, specs
from open_instruct.miles.configuration.config import CoreConfig, RunConfig


def configured():
    return RunConfig(
        CoreConfig(
            sequence_packing=True,
            attention_backend="flash_4",
            expert_parallel_size=8,
            router_aux_loss_weight=0,
            router_z_loss_weight=0,
        ),
        dict(
            hf_checkpoint="/fixture",
            actor_num_gpus_per_node=8,
            global_batch_size=8,
            rollout_batch_size=4,
            n_samples_per_prompt=2,
            context_parallel_size=2,
        ),
    )


def test_cp2_preserves_ep8_and_selects_native_miles_contiguous_response_path():
    config = configured()
    config.validate()
    assert config.resolved_miles()["allgather_cp"] is True
    assert config.core.expert_parallel_size == 8


def test_smoke_compiles_cp_without_changing_inference_tp():
    config = specs.load(Path(__file__).parents[1] / "tests/miles/configs/cp2-smoke.toml").compile()
    assert config.miles["context_parallel_size"] == 2
    assert config.miles["rollout_num_gpus_per_engine"] == 1
    assert config.core.expert_parallel_size == 2
    assert config.resolved_miles()["allgather_cp"]


def test_native_validation_keeps_replay_and_cp_response_layout():
    config = configured()
    args = SimpleNamespace(**(config.miles | {"use_rollout_routing_replay": True, "allgather_cp": True}))
    fields = constraints.native_values(args, set())
    assert fields["use_rollout_routing_replay"]
    assert fields["allgather_cp"]
    RunConfig(dataclasses.replace(config.core, replay_diagnostics=True), fields).validate()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"context_parallel_size": 3}, "CP1, CP2 or CP4"),
        ({"allgather_cp": False}, "allgather_cp"),
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({"advantage_estimator": "gspo"}, "GRPO"),
        ({"custom_loss_function_path": "fixture.loss"}, "custom loss"),
        ({"normalize_advantages": True}, "advantage whitening"),
    ],
)
def test_unqualified_parallel_contracts_fail_offline(change, match):
    config = configured()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(config, miles=config.miles | change).validate()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"sequence_packing": False}, "sequence_packing"),
        ({"use_reduce_scatter": True}, "reduce_scatter"),
        ({"router_aux_loss_weight": 0.01}, "zero router"),
    ],
)
def test_cp_requires_qualified_core_options(change, match):
    config = configured()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(config, core=dataclasses.replace(config.core, **change)).validate()
