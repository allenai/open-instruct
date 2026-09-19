"""The public filtering switch must reach the native rollout path in every mode."""

import dataclasses
from pathlib import Path

import pytest

from open_instruct.miles.config import ZERO_STD_FILTER, CoreConfig, RunConfig
from open_instruct.miles.run_spec import RunSpec


def configured(mode="barrier", enabled=True, **miles):
    return RunConfig(
        CoreConfig(publication_mode=mode, max_policy_lag=2, filter_zero_std_groups=enabled),
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
        )
        | miles,
    )


@pytest.mark.parametrize("mode", ["barrier", "engine_drain", "refresh"])
@pytest.mark.parametrize("enabled", [True, False])
def test_native_arguments_and_plan_agree(mode, enabled):
    config = configured(mode, enabled)
    argv = config.arguments()
    plan = config.plan()
    assert plan["core"]["filter_zero_std_groups"] is enabled
    if enabled:
        assert argv[argv.index("--dynamic-sampling-filter-path") + 1] == ZERO_STD_FILTER
        assert plan["miles"]["dynamic_sampling_filter_path"] == ZERO_STD_FILTER
    else:
        assert "--dynamic-sampling-filter-path" not in argv
        assert "dynamic_sampling_filter_path" not in plan["miles"]
    # Resolution must not mutate the caller's options or interfere with an opt-out.
    assert "dynamic_sampling_filter_path" not in config.miles


def test_raw_and_structured_defaults_and_researcher_override(tmp_path):
    assert CoreConfig().filter_zero_std_groups is True
    payload = {
        "schema_version": 1,
        "name": "filtering",
        "model": {"source": "/fixture/hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 8, "eval_count": 4}]},
    }
    run = RunSpec.from_dict(payload, config_path=tmp_path / "run.toml")
    assert run.compile().plan()["miles"]["dynamic_sampling_filter_path"] == ZERO_STD_FILTER
    assert run.compile().miles["n_samples_per_eval_prompt"] == 1
    disabled = RunSpec.from_dict(
        payload | {"training": {"filter_zero_std_groups": False}}, config_path=tmp_path / "run.toml"
    )
    assert "--dynamic-sampling-filter-path" not in disabled.compile().arguments()
    with pytest.raises(ValueError, match="[Cc]onflict"):
        RunSpec.from_dict(
            payload | {"training": {"filter_zero_std_groups": False}, "core": {"filter_zero_std_groups": True}},
            config_path=tmp_path / "run.toml",
        )


def test_single_response_requires_explicit_opt_out():
    with pytest.raises(ValueError, match="samples_per_prompt > 1"):
        configured(n_samples_per_prompt=1, rollout_batch_size=4).validate()
    configured(enabled=False, n_samples_per_prompt=1, rollout_batch_size=4).validate()


def test_native_filter_conflicts_are_not_silently_overridden():
    configured(dynamic_sampling_filter_path=ZERO_STD_FILTER).validate()
    with pytest.raises(ValueError, match="conflicts"):
        configured(enabled=False, dynamic_sampling_filter_path=ZERO_STD_FILTER).validate()
    with pytest.raises(ValueError, match="custom filter"):
        configured(dynamic_sampling_filter_path="custom.filter").validate()
    configured(enabled=False, dynamic_sampling_filter_path="custom.filter").validate()
    for mode in ("refresh", "engine_drain"):
        with pytest.raises(ValueError, match="only the built-in"):
            configured(mode, enabled=False, dynamic_sampling_filter_path="custom.filter").validate()


@pytest.mark.parametrize("value", ["true", 1, None])
def test_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        dataclasses.replace(CoreConfig(), filter_zero_std_groups=value)


@pytest.mark.parametrize("name", ["dev", "small", "medium", "large"])
def test_maintained_examples_explain_their_filtering_choice(name):
    path = Path(__file__).resolve().parents[1] / "configs/miles/examples" / f"{name}.toml"
    run = RunSpec.load(path)
    enabled = name in ("medium", "large")
    assert run.sections["training"]["filter_zero_std_groups"] is enabled
    assert run.compile().core.filter_zero_std_groups is enabled
    assert ("--dynamic-sampling-filter-path" in run.compile().arguments()) is enabled
