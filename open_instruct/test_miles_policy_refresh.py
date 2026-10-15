"""Validate the explicit scope of the first policy-refresh qualification."""

import dataclasses
import json

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


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


@pytest.mark.parametrize(
    "change,match",
    [
        ({"fully_async": False}, "fully_async"),
        ({"use_tis": False}, "use_tis"),
        ({"use_rollout_logprobs": True}, "use_rollout_logprobs"),
        ({"use_miles_router": False}, "use_miles_router"),
        ({"rollout_num_gpus_per_engine": 2}, "TP1"),
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
