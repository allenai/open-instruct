"""Profile transfer boundaries and user override behavior, without a GPU."""

import copy
import json
import sys

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.configuration import inference_profiles as profiles
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import launch


def model():
    return copy.deepcopy(profiles.MODEL_SHAPE) | {"dtype": "bfloat16"}


def options():
    return {"colocate": False, "rollout_max_context_len": 34816, "rollout_max_response_len": 32768}


@pytest.mark.parametrize("cluster,running", [("ai2/holmes", 512), ("ai2/jupiter", 320)])
def test_matched_candidate_and_bounded_h100_pool(cluster, running):
    report = profiles.resolve("auto", cluster, model(), options() | {"sglang_disable_radix_cache": False})
    assert report["status"] == "candidate"
    assert report["defaults"]["sglang_max_running_requests"] == running
    assert report["defaults"]["sglang_max_mamba_cache_size"] == 6 * running
    assert report["defaults"]["sglang_max_total_tokens"] == (512 if running == 512 else 128) * 34816
    assert "async_max_concurrent_samples" not in report["defaults"]


@pytest.mark.parametrize(
    "change",
    [
        {"colocate": True},
        {"rollout_num_gpus_per_engine": 2},
        {"sglang_ep_size": 2},
        {"rollout_max_context_len": 65536},
        {"rollout_max_response_len": 2048},
        {"sglang_dtype": "float16"},
        {"sglang_quantization": "fp8"},
        {"rollout_external_engine_addrs": ["localhost:1234"]},
    ],
)
def test_different_topology_or_workload_does_not_inherit_large_pools(change):
    report = profiles.resolve("auto", "ai2/holmes", model(), options() | change)
    assert report["selected"] == "generic" and report["defaults"] == {}


@pytest.mark.parametrize("key,value", [("hidden_size", 2048), ("n_routed_experts", 1024), ("dtype", "float32")])
def test_model_family_name_is_not_enough(key, value):
    report = profiles.resolve("auto", "ai2/holmes", model() | {key: value}, options())
    assert report["selected"] == "generic"


def test_unknown_hardware_missing_model_and_named_profile_mismatch():
    assert profiles.resolve("auto", "unknown", model(), options())["selected"] == "generic"
    assert profiles.resolve("auto", "ai2/holmes", None, options())["status"] == "pending_model_metadata"
    with pytest.raises(ValueError, match="conflicts"):
        profiles.resolve("olmo35-b300-v1", "ai2/jupiter", model(), options())
    with pytest.raises(ValueError, match="geometry"):
        profiles.resolve("olmo35-b300-v1", "ai2/holmes", {"model_type": "other"}, options())


def test_h100_cache_disabled_candidate_and_explicit_admission():
    report = profiles.resolve("auto", "ai2/jupiter", model(), options())
    defaults = report["defaults"]
    assert report["selected"] == "olmo35-h100-v1"
    assert defaults["sglang_max_running_requests"] == defaults["sglang_server_concurrency"] == 512
    assert defaults["sglang_max_total_tokens"] == 6684672
    assert defaults["sglang_max_mamba_cache_size"] == 512
    assert defaults["sglang_mem_fraction_static"] == 0.9
    smaller = profiles.resolve("auto", "ai2/jupiter", model(), options() | {"sglang_max_running_requests": 128})[
        "defaults"
    ]
    assert smaller["sglang_max_total_tokens"] == 128 * 34816
    assert smaller["sglang_max_mamba_cache_size"] == smaller["sglang_cuda_graph_max_bs_decode"] == 128


def run_spec(tmp_path, overrides=()):
    source = tmp_path / "model"
    source.mkdir(exist_ok=True)
    (source / "config.json").write_text(json.dumps(model()))
    return RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "profile-test",
            "model": {"source": str(source)},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
            "launch": {"cluster": "ai2/holmes"},
            "inference": {"gpus": 6, "max_context_length": 34816, "max_response_length": 32768},
            "async": {"fully_async": True},
        },
        config_path=tmp_path / "run.toml",
        overrides=list(overrides),
    )


def test_compile_profiles_preserve_global_budget_and_explicit_overrides(tmp_path):
    run = run_spec(tmp_path)
    config = run.compile()
    assert config.miles["sglang_max_running_requests"] == 512
    assert config.miles["sglang_max_mamba_cache_size"] == 3072
    assert config.miles["sglang_disable_radix_cache"] is False
    assert config.miles["sglang_mamba_radix_cache_strategy"] == "extra_buffer"
    assert config.miles["async_max_concurrent_samples"] == 768
    assert config.core.max_policy_lag == 6
    assert config.plan()["inference_profile"]["values"]["sglang_server_concurrency"]["source"] == "olmo35-b300-v1"
    overridden = run_spec(
        tmp_path,
        [
            "inference.sglang_max_running_requests=256",
            "inference.sglang_max_total_tokens=1000000",
            "async.async_max_concurrent_samples=512",
        ],
    ).compile()
    assert overridden.miles["sglang_max_running_requests"] == 256
    assert overridden.miles["sglang_cuda_graph_max_bs_decode"] == 256
    assert overridden.miles["sglang_max_mamba_cache_size"] == 1536
    assert overridden.miles["sglang_max_total_tokens"] == 1000000
    assert overridden.miles["async_max_concurrent_samples"] == 512
    assert (
        overridden.inference_profile["values"]["sglang_max_total_tokens"]["source"]
        == "inference.sglang_max_total_tokens"
    )
    restored = RunSpec.from_dict(run.to_dict(), config_path=tmp_path / "copy.json")
    assert restored.compile().miles == config.miles


@pytest.mark.parametrize("cluster,cache_off,slots", [("ai2/holmes", False, 3072), ("ai2/jupiter", True, 512)])
def test_cache_mode_is_selected_with_hardware_and_explicit_values_win(tmp_path, cluster, cache_off, slots):
    config = run_spec(tmp_path, [f'launch.cluster="{cluster}"']).compile()
    assert config.miles["sglang_disable_radix_cache"] is cache_off
    assert config.miles["sglang_max_mamba_cache_size"] == slots
    assert config.inference_profile["values"]["sglang_disable_radix_cache"]["source"].startswith("olmo35-")
    overridden = run_spec(
        tmp_path, [f'launch.cluster="{cluster}"', f"inference.radix_cache={str(cache_off).lower()}"]
    ).compile()
    assert overridden.miles["sglang_disable_radix_cache"] is not cache_off
    assert overridden.miles["sglang_max_mamba_cache_size"] == (1024 if cluster == "ai2/holmes" else 1920)
    assert overridden.inference_profile["values"]["sglang_disable_radix_cache"]["source"] == "inference.radix_cache"


def test_generic_profile_keeps_cache_disabled_and_explicit_native_override_wins(tmp_path):
    config = run_spec(tmp_path, ['inference.profile="generic"']).compile()
    assert config.miles["sglang_disable_radix_cache"] is True
    config = run_spec(tmp_path, ["miles.sglang_disable_radix_cache=true"]).compile()
    assert config.miles["sglang_disable_radix_cache"] is True
    assert config.miles["sglang_max_mamba_cache_size"] == 1024
    assert (
        config.inference_profile["values"]["sglang_disable_radix_cache"]["source"]
        == "miles.sglang_disable_radix_cache"
    )


@pytest.mark.parametrize("cluster,slots", [("ai2/holmes", 4096), ("ai2/jupiter", 2048)])
def test_explicit_cache_pool_uses_resolved_profile_strategy(tmp_path, cluster, slots):
    config = run_spec(
        tmp_path,
        [
            f'launch.cluster="{cluster}"',
            "inference.radix_cache=true",
            f"inference.sglang_max_mamba_cache_size={slots}",
        ],
    ).compile()
    assert config.miles["sglang_max_mamba_cache_size"] == slots
    assert config.miles["sglang_mamba_radix_cache_strategy"] == "extra_buffer"
    with pytest.raises(ValueError, match="must exceed"):
        run_spec(
            tmp_path,
            [f'launch.cluster="{cluster}"', "inference.radix_cache=true", "inference.sglang_max_mamba_cache_size=1"],
        ).compile()


def test_generic_selector_and_prepared_metadata(tmp_path):
    run = run_spec(tmp_path, ['inference.profile="generic"'])
    assert run.compile().miles["sglang_max_running_requests"] == 64
    run = run_spec(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "config.json").write_text(json.dumps(model() | {"hidden_size": 2048}))
    assert run.compile({"hf_checkpoint": str(prepared)}).miles["sglang_max_running_requests"] == 64


def test_hardware_check_uses_allocated_devices_and_rejects_wrong_gpu(monkeypatch):
    report = profiles.resolve("auto", "ai2/holmes", model(), options())
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return "NVIDIA B300, 275040\n"

    monkeypatch.setattr(profiles.subprocess, "check_output", query)
    assert profiles.check_hardware(report, ["2"]) == [("NVIDIA B300", 275040.0)]
    assert calls[0][0][-1] == "--id=2"
    assert calls[0][1]["timeout"] == 15
    with pytest.raises(ValueError, match="expects B300"):
        profiles.validate_hardware(report, [("NVIDIA H100", 81559)])
    with pytest.raises(ValueError, match="expects B300"):
        profiles.validate_hardware(report, [("NVIDIA B300", 100000)])
    assert profiles.check_hardware({"status": "fallback"}) == []
    assert len(calls) == 1


def warning_map(config):
    return {warning["code"]: warning for warning in config.plan()["throughput"]["warnings"]}


def test_fleet_extrapolation_is_independent_of_capacity_warnings(tmp_path):
    config = run_spec(tmp_path, ["inference.gpus=78", "async.async_max_concurrent_samples=39936"]).compile()
    warning = warning_map(config)["inference_fleet_extrapolation"]
    assert warning["requested_engines"] == 78
    assert "publication" in warning["message"]
    assert "inference_profile_modified" not in warning_map(config)
    assert config.miles["sglang_max_running_requests"] == 512


@pytest.mark.parametrize("selector", ["auto", "olmo35-b300-v1"])
def test_matching_single_engine_and_equal_overrides_do_not_warn(tmp_path, selector):
    config = run_spec(
        tmp_path, [f'inference.profile="{selector}"', "inference.gpus=1", "inference.sglang_max_running_requests=512"]
    ).compile()
    assert config.inference_profile["warnings"] == []


def test_generic_matching_profile_warns_without_changing_settings(tmp_path):
    config = run_spec(tmp_path, ['inference.profile="generic"']).compile()
    warning = warning_map(config)["inference_profile_bypassed"]
    assert warning["differences"]["sglang_max_running_requests"] == {"configured": 64, "profile_default": 512}
    assert config.miles["sglang_disable_radix_cache"] is True
    assert config.inference_profile["selected"] == "generic"
    unknown = run_spec(tmp_path, ['inference.profile="generic"', 'launch.cluster="ai2/saturn"']).compile()
    assert "inference_profile_bypassed" not in warning_map(unknown)


def test_derived_and_explicit_profile_departures_are_reported(tmp_path):
    config = run_spec(tmp_path, ["inference.sglang_max_running_requests=256"]).compile()
    differences = warning_map(config)["inference_profile_modified"]["differences"]
    assert differences["sglang_max_running_requests"]["configured"] == 256
    assert differences["sglang_cuda_graph_max_bs_decode"]["configured"] == 256
    assert differences["sglang_max_mamba_cache_size"]["configured"] == 1536
    assert config.miles["sglang_max_running_requests"] == 256


def test_fallback_and_pending_metadata_are_visible_and_rechecked(tmp_path):
    fallback = run_spec(tmp_path, ["inference.rollout_tensor_parallel_size=2"]).compile()
    assert "dedicated TP1" in warning_map(fallback)["inference_profile_fallback"]["message"]
    run = run_spec(tmp_path)
    (tmp_path / "model" / "config.json").unlink()
    assert "inference_profile_pending" in warning_map(run.compile())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "config.json").write_text(json.dumps(model()))
    assert "inference_profile_pending" not in warning_map(run.compile({"hf_checkpoint": str(prepared)}))
    generic = run_spec(tmp_path, ['inference.profile="generic"'])
    (tmp_path / "model" / "config.json").write_text("invalid json")
    assert "inference_profile_pending" in warning_map(generic.compile())
    assert "inference_profile_bypassed" in warning_map(generic.compile({"hf_checkpoint": str(prepared)}))


@pytest.mark.parametrize("command", ["plan", "validate"])
def test_cli_surfaces_advisories_and_keeps_plan_json_clean(tmp_path, monkeypatch, capsys, command):
    run = run_spec(tmp_path, ['inference.profile="generic"'])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run.to_dict()))
    monkeypatch.setattr(sys, "argv", ["miles", command, str(path)])
    cli.main()
    captured = capsys.readouterr()
    assert "Warning [inference_profile_bypassed]" in captured.err
    assert "Warning [inference_fleet_extrapolation]" in captured.err
    if command == "plan":
        plan = json.loads(captured.out)
        assert plan["runtime"]["inference_profile"]["selected"] == "generic"


def test_submission_surfaces_advisories_before_build(tmp_path, monkeypatch, capsys):
    run = run_spec(tmp_path, ['inference.profile="generic"'])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run.to_dict()))
    builds = []

    def build(command, **kwargs):
        captured = capsys.readouterr()
        assert "Warning [inference_profile_bypassed]" in captured.err
        assert "Warning [inference_fleet_extrapolation]" in captured.err
        builds.append(command)

    monkeypatch.setattr(launch.subprocess, "run", build)
    launch.run(path, [])
    assert len(builds) == 1
    assert "--miles" in builds[0]
