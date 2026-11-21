"""Validate the explicit scope of the first policy-refresh qualification."""

import dataclasses
import json
from pathlib import Path

import pytest

from open_instruct.miles.configuration import specs, topology
from open_instruct.miles.configuration.config import EXCLUDE_TRUNCATED, EXCLUDE_UNGRADED, CoreConfig, RunConfig


def configured():
    return RunConfig(
        CoreConfig(publication_mode="refresh", max_policy_lag=2),
        dict(
            hf_checkpoint="/fixture/hf",
            global_batch_size=4,
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            fully_async=True,
            use_miles_router=True,
            use_tis=True,
            sglang_cuda_graph_backend_decode="disabled",
            sglang_cuda_graph_backend_prefill="disabled",
        ),
    )


def test_refresh_is_explicit_and_compiles_into_core_config():
    config = configured()
    config.validate()
    assert '"publication_mode": "refresh"' in config.arguments()[-1]
    assert CoreConfig().publication_mode == "barrier"


def test_tp2_refresh_smoke_compiles_one_engine_and_two_trainer_ranks():
    spec = specs.load(Path(__file__).parents[1] / "tests/miles/configs/tp2-refresh.toml")
    config = spec.compile()
    assert config.core.publication_mode == "refresh"
    assert config.miles["rollout_num_gpus_per_engine"] == 2
    assert config.miles["rollout_num_gpus"] == 2
    assert config.miles["actor_num_gpus_per_node"] == 2
    assert config.miles["sglang_cuda_graph_backend_decode"] == "full"
    assert config.miles["sglang_cuda_graph_max_bs_decode"] == config.miles["sglang_max_running_requests"] == 8
    assert config.miles["sglang_cuda_graph_backend_prefill"] == "disabled"
    assert topology.plan(spec, config)["replicas"] == 1


@pytest.mark.parametrize(
    "change,match",
    [
        ({"fully_async": False}, "fully_async"),
        ({"use_tis": False}, "use_tis"),
        ({"use_rollout_logprobs": True}, "use_rollout_logprobs"),
        ({"rollout_num_gpus_per_engine": 4}, "TP1 or TP2"),
        ({"use_fault_tolerance": True}, "use_fault_tolerance"),
        ({"rollout_temperature": 0.7}, "temperature"),
        ({"rollout_top_p": 0.9}, "top_p"),
        ({"rollout_top_k": 10}, "top_k"),
        ({"sglang_cuda_graph_backend_decode": "tc_piecewise"}, "backend_decode"),
        ({"sglang_cuda_graph_backend_prefill": "tc_piecewise"}, "backend_prefill"),
        ({"advantage_estimator": "gspo"}, "grpo"),
    ],
)
def test_refresh_rejects_unqualified_contracts(change, match):
    config = configured()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(config, miles=config.miles | change).validate()


@pytest.mark.parametrize("decode", ["disabled", "full"])
def test_refresh_supports_tp2_without_expanding_engine_drain_scope(decode):
    config = configured()
    config = dataclasses.replace(
        config, miles=config.miles | {"rollout_num_gpus_per_engine": 2, "sglang_cuda_graph_backend_decode": decode}
    )
    config.validate()
    with pytest.raises(ValueError, match="TP1 serving"):
        dataclasses.replace(config, core=dataclasses.replace(config.core, publication_mode="engine_drain")).validate()


def test_refresh_full_decode_graphs_preserve_prefill_restriction():
    config = configured()
    config = dataclasses.replace(config, miles=config.miles | {"sglang_cuda_graph_backend_decode": "full"})
    config.validate()
    with pytest.raises(ValueError, match="backend_prefill"):
        dataclasses.replace(config, miles=config.miles | {"sglang_cuda_graph_backend_prefill": "breakable"}).validate()


@pytest.mark.parametrize("encode", [lambda value: value, json.dumps])
def test_refresh_checks_effective_json_graph_overrides(encode):
    config = configured()
    with pytest.raises(ValueError, match="backend_prefill"):
        dataclasses.replace(
            config, miles=config.miles | {"sglang_cuda_graph_config": encode({"prefill": {"backend": "breakable"}})}
        ).validate()
    dataclasses.replace(
        config,
        miles=config.miles
        | {
            "sglang_cuda_graph_backend_decode": "tc_piecewise",
            "sglang_cuda_graph_config": encode({"decode": {"backend": "full"}}),
        },
    ).validate()


def test_refresh_min_group_size_defaults_off_and_compiles_into_core_config():
    assert CoreConfig().refresh_min_group_size is None
    assert '"refresh_min_group_size": null' in configured().arguments()[-1]
    config = configured()
    config = dataclasses.replace(
        config,
        core=dataclasses.replace(
            config.core, refresh_min_group_size=2, router_aux_loss_weight=0, router_z_loss_weight=0
        ),
        miles={**config.miles, "n_samples_per_prompt": 4, "global_batch_size": 8},
    )
    config.validate()
    assert '"refresh_min_group_size": 2' in config.arguments()[-1]


@pytest.mark.parametrize("value", [1, 0, True, 2.0, "2"])
def test_refresh_min_group_size_must_be_an_integer_of_at_least_two(value):
    with pytest.raises(ValueError, match="refresh_min_group_size"):
        CoreConfig(publication_mode="refresh", max_policy_lag=2, refresh_min_group_size=value)


def test_partial_groups_accept_only_baseline_aware_post_processes():
    config = configured()
    n = 4

    def with_post_process(path):
        miles = {**config.miles, "n_samples_per_prompt": n, "global_batch_size": 2 * n}
        if path is not None:
            miles["custom_reward_post_process_path"] = path
        core = dataclasses.replace(
            config.core,
            refresh_min_group_size=n - 1,
            reward_zero_truncated=False,
            router_aux_loss_weight=0,
            router_z_loss_weight=0,
        )
        return RunConfig(core, miles)

    for accepted in (None, EXCLUDE_UNGRADED, EXCLUDE_TRUNCATED):
        with_post_process(accepted).validate()
    with pytest.raises(ValueError, match="requires the built-in reward post-process"):
        with_post_process("custom.post_process").validate()
    # Full groups keep any post-process.
    miles = {
        **config.miles,
        "n_samples_per_prompt": n,
        "global_batch_size": 2 * n,
        "custom_reward_post_process_path": "custom.post_process",
    }
    RunConfig(
        dataclasses.replace(config.core, refresh_min_group_size=n, reward_zero_truncated=False), miles
    ).validate()


def test_refresh_min_group_size_requires_refresh_and_fits_the_group():
    config = configured()
    with pytest.raises(ValueError, match="cannot exceed"):
        dataclasses.replace(config, core=dataclasses.replace(config.core, refresh_min_group_size=3)).validate()
    with pytest.raises(ValueError, match="publication_mode=refresh"):
        RunConfig(CoreConfig(refresh_min_group_size=2), dict(config.miles)).validate()


@pytest.mark.parametrize("weight", ["router_aux_loss_weight", "router_z_loss_weight"])
def test_partial_groups_reject_router_losses(weight):
    config = configured()
    weights = dict(router_aux_loss_weight=0.0, router_z_loss_weight=0.0)
    weights[weight] = 0.1
    core = dataclasses.replace(config.core, refresh_min_group_size=2, **weights)
    with pytest.raises(ValueError, match=rf"Experimental partial refresh groups require .*{weight}=0"):
        RunConfig(core, {**config.miles, "n_samples_per_prompt": 4, "global_batch_size": 8}).validate()
