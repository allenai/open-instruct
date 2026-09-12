"""Structured radix-cache and router controls resolve to native options and are capacity-checked."""

from pathlib import Path

import pytest

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec

EXAMPLE = Path(__file__).resolve().parents[2] / "configs/miles/examples/grpo-async-disaggregated.toml"

RADIX_ON = [
    "inference.radix_cache=true",
    'inference.mamba_radix_cache_strategy="extra_buffer"',
    "inference.sglang_max_mamba_cache_size=332",
    'inference.router_policy="cache_aware"',
    "inference.router_cache_threshold=0.8",
    "inference.router_balance_abs_threshold=4",
    "inference.router_balance_rel_threshold=1.5",
]


def _miles(overrides):
    return RunSpec.load(EXAMPLE, overrides=overrides).compile().miles


def test_example_default_is_cache_off_with_a_slot_per_request():
    miles = _miles([])
    assert miles["sglang_disable_radix_cache"] is True
    assert miles["sglang_max_mamba_cache_size"] >= miles["sglang_max_running_requests"]


def test_radix_controls_resolve_to_native_names():
    miles = _miles(RADIX_ON + ["inference.enable_mixed_chunk=true"])
    assert miles["sglang_disable_radix_cache"] is False
    assert miles["sglang_mamba_radix_cache_strategy"] == "extra_buffer"
    assert miles["sglang_max_mamba_cache_size"] == 332
    assert miles["sglang_router_policy"] == "cache_aware"
    assert miles["router_cache_threshold"] == 0.8
    assert miles["router_balance_abs_threshold"] == 4
    assert miles["router_balance_rel_threshold"] == 1.5
    assert miles["sglang_enable_mixed_chunk"] is True


@pytest.mark.parametrize(
    "override, message",
    [
        ("inference.sglang_max_mamba_cache_size=320", "must exceed 5"),
        ('inference.mamba_radix_cache_strategy="auto"', "extra_buffer"),
        ('inference.sglang_attention_backend="flashinfer"', "triton"),
        ("inference.sglang_page_size=16", "page_size = 1"),
        ("inference.sglang_disable_overlap_schedule=true", "overlap scheduling"),
        ("inference.router_cache_threshold=1.5", "router_cache_threshold"),
        ("inference.router_balance_rel_threshold=0.5", "router_balance_rel_threshold"),
    ],
)
def test_radix_capacity_rules_reject_unvalidated_settings(override, message):
    with pytest.raises(InputError, match=message):
        _miles(RADIX_ON + [override])


def test_cache_off_still_needs_a_slot_per_running_request():
    with pytest.raises(InputError, match="at least sglang_max_running_requests"):
        _miles(["inference.sglang_max_mamba_cache_size=32"])


def test_structured_and_native_names_cannot_disagree():
    with pytest.raises(InputError, match="Conflicting"):
        _miles(RADIX_ON + ['inference.sglang_router_policy="round_robin"'])


@pytest.mark.parametrize(
    "values",
    [{}, {"sglang_disable_radix_cache": False, "sglang_attention_backend": "flashinfer", "sglang_page_size": 16}],
)
def test_dense_and_unspecified_caches_do_not_require_kda_controls(values):
    validation.inference_capacity(values)


def test_explicit_recurrent_pool_still_requires_qualified_radix_strategy():
    with pytest.raises(InputError, match="extra_buffer"):
        validation.inference_capacity({"sglang_disable_radix_cache": False, "sglang_max_mamba_cache_size": 332})
