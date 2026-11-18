"""Compile the owned cache backend through the unchanged pinned SGLang options."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration import inference_profiles as profiles
from open_instruct.miles.configuration import options, validation
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/miles/examples/tiny/node.toml"
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
        (["miles.sglang_tp_size=2"], "TP1/PP1/DP1/EP1"),
        (["miles.sglang_pp_size=2"], "TP1/PP1/DP1/EP1"),
        (["miles.sglang_ep_size=2"], "TP1/PP1/DP1/EP1"),
        (["miles.rollout_num_gpus_per_engine=2"], "single-GPU"),
        (['miles.pause_generation_mode="retract"'], "requires non-colocated serving"),
        (['inference.placement_mode="colocated"', "inference.gpus=2"], "requires non-colocated serving"),
    ],
)
def test_unsafe_combinations_remain_rejected(extra, match):
    with pytest.raises(InputError, match=match):
        compile_config(extra)


def test_existing_example_still_disables_cache():
    config = RunSpec.load(EXAMPLE).compile()
    assert config.miles["sglang_disable_radix_cache"] is True
    assert config.miles.get("sglang_radix_cache_backend") is None


@pytest.mark.parametrize(
    "key,native",
    [
        ("sglang_max_running_requests", "sglang_max_running_requests"),
        ("sglang_max_mamba_cache_size", "sglang_max_mamba_cache_size"),
        ("mamba_radix_cache_strategy", "sglang_mamba_radix_cache_strategy"),
    ],
)
def test_policy_cache_requires_explicit_capacity_before_defaults(key, native):
    overrides = [value for value in OVERRIDES if not value.startswith(f"inference.{key}=")]
    with pytest.raises(InputError, match=f"requires explicit capacity settings: {native}"):
        RunSpec.load(EXAMPLE, overrides=overrides).compile()


@pytest.mark.parametrize("profile", profiles.PROFILES)
def test_policy_cache_cannot_select_measured_presets(profile):
    with pytest.raises(InputError, match="Policy-versioned caching is experimental"):
        compile_config([f'inference.profile="{profile}"'])


def test_advisor_does_not_recommend_cache_off_presets_for_policy_cache():
    config = compile_config()
    model = profiles.MODEL_SHAPE | {"dtype": "bfloat16"}
    _, tips = profiles.advise(config.inference_profile, "ai2/holmes", model, config.miles, set(config.miles))
    assert not any(tip["code"] == "inference_profile_available" for tip in tips)
