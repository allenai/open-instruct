"""Profile transfer boundaries and user override behavior, without a GPU."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.configuration import advisories
from open_instruct.miles.configuration import inference_profiles as profiles
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import launch, workflow
from open_instruct.miles.infrastructure import gpu_hardware


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


def test_compile_profiles_size_global_budget_and_preserve_explicit_overrides(tmp_path):
    run = run_spec(tmp_path)
    config = run.compile()
    assert config.miles["sglang_max_running_requests"] == 512
    assert config.miles["sglang_max_mamba_cache_size"] == 3072
    assert config.miles["sglang_disable_radix_cache"] is False
    assert config.miles["sglang_mamba_radix_cache_strategy"] == "extra_buffer"
    assert config.miles["async_max_concurrent_samples"] == 6144
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
        return "2, NVIDIA B300, 275040\n"

    monkeypatch.setattr(gpu_hardware.subprocess, "check_output", query)
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
    return {warning["code"]: warning for warning in config.plan()["inference_profile"]["warnings"]}


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
    fallback = run_spec(tmp_path, ['inference.profile="auto"', "inference.rollout_tensor_parallel_size=2"]).compile()
    assert "dedicated TP1" in warning_map(fallback)["inference_profile_fallback"]["message"]
    run = run_spec(tmp_path, ['inference.profile="auto"'])
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


@pytest.mark.parametrize(
    "change",
    [
        {"colocate": True},
        {"rollout_num_gpus_per_engine": 2},
        {"rollout_external_engine_addrs": ["host:1234"]},
        {"sglang_ep_size": 2},
        {"sglang_quantization": "fp8"},
        {"sglang_dtype": "float16"},
        {"rollout_max_context_len": 65536},
    ],
)
def test_named_profile_rejects_known_conflicts_without_model_metadata(change):
    with pytest.raises(ValueError):
        profiles.resolve("olmo35-b300-v1", "ai2/holmes", None, options() | change)


def test_named_hardware_conflict_without_metadata_and_valid_pending():
    with pytest.raises(ValueError, match="conflicts"):
        profiles.resolve("olmo35-b300-v1", "ai2/jupiter", None, options())
    with pytest.raises(ValueError, match="conflicts"):
        profiles.resolve("olmo35-b300-v1", "ai2/saturn", None, options())
    assert profiles.resolve("olmo35-b300-v1", "ai2/holmes", None, options())["status"] == "pending_model_metadata"


@pytest.mark.parametrize("gpus,admission,expected", [(1, 512, 1024), (6, 512, 6144), (19, 512, 19456), (6, 256, 3072)])
def test_automatic_producer_uses_resolved_capacity(tmp_path, gpus, admission, expected):
    config = run_spec(
        tmp_path, [f"inference.gpus={gpus}", f"inference.sglang_max_running_requests={admission}"]
    ).compile()
    assert config.miles["async_max_concurrent_samples"] == expected
    assert not any(
        "below" in warning and "Producer budget" in warning for warning in config.plan()["async_capacity"]["warnings"]
    )
    assert config.core.max_policy_lag == 6


def test_explicit_underfeeding_budget_is_preserved_and_omitting_it_fixes_capacity(tmp_path):
    config = run_spec(tmp_path, ["async.async_max_concurrent_samples=768"]).compile()
    assert config.miles["async_max_concurrent_samples"] == 768
    assert any(
        "Producer budget 768 is below 3072" in warning for warning in config.plan()["async_capacity"]["warnings"]
    )
    assert run_spec(tmp_path).compile().miles["async_max_concurrent_samples"] == 6144


def test_implicit_unmatched_auto_is_quiet_but_explicit_auto_warns(tmp_path):
    run = run_spec(tmp_path)
    (tmp_path / "model" / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert run.compile().inference_profile["warnings"] == []
    explicit = run_spec(tmp_path, ['inference.profile="auto"', 'launch.cluster="ai2/saturn"']).compile()
    assert "inference_profile_fallback" in warning_map(explicit)


def test_commands_emit_same_warnings_without_duplicate_json_entries(tmp_path, monkeypatch, capsys):
    run = run_spec(tmp_path, ['inference.profile="generic"', "async.async_max_concurrent_samples=8"])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run.to_dict()))
    messages = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *args, **kwargs: None)
    for command in ("plan", "validate", "run"):
        monkeypatch.setattr(sys, "argv", ["miles", command, str(path)])
        cli.main()
        captured = capsys.readouterr()
        messages.append(captured.err)
        if command == "plan":
            plan = json.loads(captured.out)["runtime"]
            assert not any(w["code"].startswith("inference_") for w in plan["throughput"]["warnings"])
            assert captured.out.count('"code": "inference_profile_bypassed"') == 1
    assert messages[0] == messages[1] == messages[2]
    assert "Producer budget 8" in messages[0]


def test_submission_reuses_one_compilation_after_loading(tmp_path, monkeypatch):
    run = run_spec(tmp_path)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run.to_dict()))
    original = RunSpec.compile
    calls = []

    def compile_once(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(launch.specs, "load", lambda *args: run)
    monkeypatch.setattr(RunSpec, "compile", compile_once)
    monkeypatch.setattr(launch.subprocess, "run", lambda *args, **kwargs: None)
    launch.run(path, [])
    assert len(calls) == 1


def test_shared_gpu_inventory_preserves_mask_order_and_unknown_memory(monkeypatch):
    monkeypatch.setattr(
        gpu_hardware.subprocess,
        "check_output",
        lambda *args, **kwargs: "2, NVIDIA B300, 275040\n0, NVIDIA H100, N/A\n",
    )
    inventory = gpu_hardware.inventory(["0", "2"])
    assert [device["id"] for device in inventory] == ["0", "2"]
    assert inventory[0]["memory_mib"] is None
    report = profiles.resolve("auto", "ai2/jupiter", model(), options())
    with pytest.raises(ValueError, match="expects H100"):
        profiles.validate_hardware(report, [("NVIDIA H100", None)])
    with pytest.raises(ValueError, match="MIG/UUID"):
        gpu_hardware.inventory(["MIG-example"])


def test_rollout_inventory_excludes_trainers_and_judges():
    devices = [
        {"id": "3", "name": "NVIDIA H100", "memory_mib": 81559},
        {"id": "1", "name": "NVIDIA B300", "memory_mib": 275040},
        {"id": "2", "name": "NVIDIA A100", "memory_mib": 81920},
    ]
    report = gpu_hardware.rollout_inventory({"trainer_gpus": 1, "rollout_gpus": 1}, devices)
    assert report["device_ids"] == ["1"]
    assert report["hardware"] == [("NVIDIA B300", 275040)]


def test_prepared_profile_checks_every_replica_even_when_startup_was_pending(tmp_path):
    pending = profiles.resolve("auto", "ai2/holmes", None, options())
    layout = {"replicas": 3, "nodes": [{"rollout_gpus": 0}, {"rollout_gpus": 1}, {"rollout_gpus": 1}]}
    for rank, devices in enumerate([[], [["NVIDIA B300", 275040]], [["NVIDIA H100", 81559]]]):
        (tmp_path / f"inference-profile-{rank}.json").write_text(
            json.dumps({"profile": pending, "rollout_gpus": len(devices), "hardware": devices})
        )
    resolved = profiles.resolve("auto", "ai2/holmes", model(), options())
    with pytest.raises(ValueError, match="inference-profile-2.*expects B300"):
        profiles.check_cluster_hardware(resolved, tmp_path, layout)
    (tmp_path / "inference-profile-2.json").write_text(
        json.dumps({"rollout_gpus": 1, "hardware": [["NVIDIA B300", 275040]]})
    )
    assert len(profiles.check_cluster_hardware(resolved, tmp_path, layout)) == 3
    (tmp_path / "inference-profile-1.json").unlink()
    with pytest.raises(ValueError, match="inference-profile-1"):
        profiles.check_cluster_hardware(resolved, tmp_path, layout)


@pytest.mark.parametrize("wrong_gpu", [False, True])
def test_workflow_validates_remote_inventory_after_preparation(tmp_path, monkeypatch, wrong_gpu):
    run = run_spec(tmp_path, ["inference.gpus=14"])
    (tmp_path / "model" / "config.json").unlink()
    assert run.compile().inference_profile["status"] == "pending_model_metadata"
    prepared = tmp_path / "prepared-hf"
    prepared.mkdir()
    (prepared / "config.json").write_text(json.dumps(model()))
    reports = tmp_path / "replicas"
    reports.mkdir()
    for rank, count in enumerate([6, 8]):
        name = "NVIDIA H100" if wrong_gpu and rank == 1 else "NVIDIA B300"
        (reports / f"inference-profile-{rank}.json").write_text(
            json.dumps({"rollout_gpus": count, "hardware": [[name, 275040]] * count})
        )
    monkeypatch.setenv("OI_MILES_INFERENCE_REPORT_DIR", str(reports))
    monkeypatch.setattr(workflow, "prepare_model", lambda spec: str(prepared))
    data = SimpleNamespace(prepare_data=lambda *args, **kwargs: {})
    registry = SimpleNamespace(validate_data=lambda *args: None)
    original_import = workflow.importlib.import_module

    def import_module(name):
        if name == "open_instruct.miles.datasets.run_data":
            return data
        if name == "open_instruct.miles.rewards.judge_registry":
            return registry
        return original_import(name)

    monkeypatch.setattr(workflow.importlib, "import_module", import_module)
    monkeypatch.setattr(
        profiles, "check_hardware", lambda *args: pytest.fail("Checked driver GPUs instead of replica inventories")
    )
    trained = []
    monkeypatch.setattr(workflow, "train_config", lambda config, **kwargs: trained.append(config))
    if wrong_gpu:
        with pytest.raises(ValueError, match="inference-profile-1.*expects B300"):
            workflow.execute(run)
        assert not trained
    else:
        assert workflow.execute(run)["status"] == "complete"
        assert len(trained) == 1
        assert trained[0].inference_profile["status"] == "candidate"


@pytest.mark.parametrize("cluster,cache", [("ai2/holmes", False), ("ai2/jupiter", True)])
def test_measured_cache_alternatives_are_reference_baselines(tmp_path, cluster, cache):
    config = run_spec(
        tmp_path, [f'launch.cluster="{cluster}"', f"inference.radix_cache={str(cache).lower()}"]
    ).compile()
    assert "inference_profile_modified" not in warning_map(config)
    assert config.inference_profile["matched_reference_defaults"]["sglang_disable_radix_cache"] is not cache


def test_measured_http_equality_is_not_advised_away_but_supply_changes_are(tmp_path):
    config = run_spec(tmp_path).compile()
    assert not any("End-to-end request capacity equals" in w for w in config.plan()["async_capacity"]["warnings"])
    limited = run_spec(tmp_path, ["async.async_max_concurrent_samples=3072"]).compile()
    assert any("End-to-end request capacity equals" in w for w in limited.plan()["async_capacity"]["warnings"])
    changed = run_spec(tmp_path, ["inference.sglang_max_running_requests=256"]).compile()
    assert any("End-to-end request capacity equals" in w for w in changed.plan()["async_capacity"]["warnings"])


def test_profile_pool_extrapolation_requires_explicit_memory_choices(tmp_path):
    overrides = [
        'launch.cluster="ai2/jupiter"',
        "inference.radix_cache=true",
        "inference.sglang_max_running_requests=512",
    ]
    with pytest.raises(ValueError, match="Automatic pool extrapolation"):
        run_spec(tmp_path, overrides)
    with pytest.raises(ValueError, match="above this profile.*static-memory budget"):
        run_spec(
            tmp_path,
            overrides + ["inference.sglang_max_total_tokens=4456448", "inference.sglang_max_mamba_cache_size=3072"],
        )
    # Explicit larger pools can pass the lower-bound check on a GPU with room;
    # this does not claim that graph/workspace peaks or throughput are qualified.
    explicit = run_spec(
        tmp_path,
        [
            "inference.sglang_max_running_requests=768",
            "inference.sglang_max_total_tokens=17825792",
            "inference.sglang_max_mamba_cache_size=4608",
        ],
    ).compile()
    assert explicit.miles["sglang_max_running_requests"] == 768
    assert "inference_profile_modified" in warning_map(explicit)


def test_single_node_hardware_check_only_uses_rollout_slice(monkeypatch):
    report = profiles.resolve("auto", "ai2/holmes", model(), options())
    monkeypatch.setattr(
        gpu_hardware,
        "inventory",
        lambda *args: [
            {"id": "0", "name": "NVIDIA H100", "memory_mib": 81559},
            {"id": "1", "name": "NVIDIA B300", "memory_mib": 275040},
            {"id": "2", "name": "NVIDIA A100", "memory_mib": 81559},
        ],
    )
    assert profiles.check_hardware(report, node={"trainer_gpus": 1, "rollout_gpus": 1}) == [("NVIDIA B300", 275040)]
    with pytest.raises(ValueError, match="expects B300"):
        profiles.check_hardware(report, node={"trainer_gpus": 0, "rollout_gpus": 1})


@pytest.mark.parametrize("mask", ["", "-1"])
def test_hidden_gpu_mask_never_becomes_all_gpus(monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        gpu_hardware.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("Queried hidden GPUs")
    )
    assert gpu_hardware.inventory() == []


def test_duplicate_startup_gpu_mask_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        gpu_hardware.inventory(["0", "0"])


@pytest.mark.parametrize(
    "plan",
    [
        {"runtime": {"async_capacity": None, "throughput": None, "inference_profile": None}},
        {"async_capacity": None},
        {"warnings": ["OPD warning"]},
    ],
)
def test_advisory_renderer_handles_supported_optional_sections(capsys, plan):
    advisories.emit(plan)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("Warning: OPD warning\n" if "warnings" in plan else "")
