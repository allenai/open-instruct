"""Compile the owned cache backend through the unchanged pinned SGLang options."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration import options, validation
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/miles/examples/node.toml"
OVERRIDES = [
    'inference.profile="generic"',
    "inference.radix_cache=true",
    'inference.mamba_radix_cache_strategy="extra_buffer"',
    "inference.sglang_max_running_requests=64",
    "inference.sglang_server_concurrency=64",
    "inference.sglang_max_mamba_cache_size=512",
    'inference.router_policy="consistent_hashing"',
    'miles.sglang_schedule_policy="lpm"',
    f'miles.sglang_radix_cache_backend="{validation.POLICY_CACHE_BACKEND}"',
]


def compile_config(extra=()):
    return RunSpec.load(EXAMPLE, overrides=OVERRIDES + list(extra)).compile()


def test_owned_backend_compiles_without_new_engine_flags():
    config = compile_config()
    assert config.miles["pause_generation_mode"] == "in_place"
    assert config.miles["sglang_disable_radix_cache"] is False
    assert config.miles["sglang_radix_cache_backend"] == validation.POLICY_CACHE_BACKEND
    encoded = options.encode_options({"sglang_radix_cache_backend": validation.POLICY_CACHE_BACKEND})
    assert encoded == ["--sglang-radix-cache-backend", validation.POLICY_CACHE_BACKEND]


@pytest.mark.parametrize(
    "extra,match",
    [
        (['miles.sglang_radix_cache_backend="ordinary"'], "in_place publication requires"),
        (["inference.radix_cache=false"], "requires inference.radix_cache=true"),
        (["inference.sglang_max_mamba_cache_size=320"], "must exceed 5"),
        (['miles.sglang_speculative_algorithm="EAGLE"'], "does not support speculative"),
        (["miles.sglang_dp_size=2"], "TP1"),
    ],
)
def test_unsafe_combinations_remain_rejected(extra, match):
    with pytest.raises(InputError, match=match):
        compile_config(extra)


def test_existing_example_still_disables_cache():
    config = RunSpec.load(EXAMPLE).compile()
    assert config.miles["sglang_disable_radix_cache"] is True
    assert config.miles.get("sglang_radix_cache_backend") is None
