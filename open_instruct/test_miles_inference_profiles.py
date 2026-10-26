"""Profile transfer boundaries and user override behavior, without a GPU."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import tomllib

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


@pytest.mark.parametrize(
    "change",
    [
        {"colocate": True},
        {"rollout_num_gpus_per_engine": 2},
        {"sglang_attention_backend": "flashinfer"},
        {"sglang_page_size": 16},
        {"sglang_disable_overlap_schedule": True},
        {"sglang_ep_size": 2},
        {"rollout_max_context_len": 65536},
        {"rollout_max_response_len": 2048},
        {"sglang_dtype": "float16"},
        {"sglang_quantization": "fp8"},
        {"rollout_external_engine_addrs": ["localhost:1234"]},
    ],
)
def test_different_topology_or_workload_does_not_inherit_large_pools(change):
    with pytest.raises(ValueError):
        profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options() | change)


@pytest.mark.parametrize("key,value", [("hidden_size", 2048), ("n_routed_experts", 1024), ("dtype", "float32")])
def test_model_family_name_is_not_enough(key, value):
    with pytest.raises(ValueError, match="geometry"):
        profiles.resolve("olmo35-b300-v1", "ai2/holmes", model() | {key: value}, options())


def test_unknown_hardware_missing_model_and_named_profile_mismatch():
    assert profiles.resolve("generic", "unknown", model(), options())["defaults"] == {}
    assert profiles.resolve("olmo35-b300-v1", "ai2/holmes", None, options())["model_metadata_available"] is False
    with pytest.raises(ValueError, match="expects B300"):
        profiles.resolve("olmo35-b300-v1", "ai2/jupiter", model(), options())
    with pytest.raises(ValueError, match="geometry"):
        profiles.resolve("olmo35-b300-v1", "ai2/holmes", {"model_type": "other"}, options())


def run_spec(tmp_path, overrides=(), *, profile="olmo35-b300-v1"):
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
            "inference": {
                "gpus": 6,
                "max_context_length": 34816,
                "max_response_length": 32768,
                **({"profile": profile} if profile is not None else {}),
            },
            "async": {"fully_async": True},
        },
        config_path=tmp_path / "run.toml",
        overrides=list(overrides),
    )


def test_hardware_check_uses_allocated_devices_and_rejects_wrong_gpu(monkeypatch):
    report = profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options())
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return "2, NVIDIA B300, 275040\n"

    monkeypatch.setattr(gpu_hardware.subprocess, "check_output", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert profiles.check_hardware(report, node={"trainer_gpus": 0, "rollout_gpus": 1}) == [("NVIDIA B300", 275040.0)]
    assert calls[0][0][-1] == "--id=2"
    assert calls[0][1]["timeout"] == 15
    with pytest.raises(ValueError, match="expects B300"):
        profiles.validate_hardware(report, [("NVIDIA H100", 81559)])
    with pytest.raises(ValueError, match="expects B300"):
        profiles.validate_hardware(report, [("NVIDIA B300", 100000)])
    assert profiles.check_hardware({"selected": "generic"}, node={"trainer_gpus": 0, "rollout_gpus": 1}) == []
    assert len(calls) == 1


def warning_map(config):
    report = config.plan()["inference_profile"]
    return {warning["code"]: warning for warning in report["warnings"] + report["tips"]}


def test_fleet_extrapolation_is_independent_of_capacity_warnings(tmp_path):
    config = run_spec(tmp_path, ["inference.gpus=78", "async.async_max_concurrent_samples=39936"]).compile()
    warning = warning_map(config)["inference_fleet_extrapolation"]
    assert warning["requested_engines"] == 78
    assert "publication" in warning["message"]
    assert "inference_profile_modified" not in warning_map(config)
    assert config.miles["sglang_max_running_requests"] == 512


@pytest.mark.parametrize(
    "change",
    [
        {"colocate": True},
        {"rollout_num_gpus_per_engine": 2},
        {"rollout_external_engine_addrs": ["host:1234"]},
        {"sglang_attention_backend": "flashinfer"},
        {"sglang_page_size": 16},
        {"sglang_disable_overlap_schedule": True},
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
    with pytest.raises(ValueError, match="expects B300"):
        profiles.resolve("olmo35-b300-v1", "ai2/jupiter", None, options())
    with pytest.raises(ValueError, match="expects B300"):
        profiles.resolve("olmo35-b300-v1", "ai2/saturn", None, options())
    assert profiles.resolve("olmo35-b300-v1", "ai2/holmes", None, options())["model_metadata_available"] is False


def test_explicit_underfeeding_budget_is_preserved_and_omitting_it_fixes_capacity(tmp_path):
    config = run_spec(tmp_path, ["async.async_max_concurrent_samples=768"]).compile()
    assert config.miles["async_max_concurrent_samples"] == 768
    assert any(
        "Producer budget 768 is below 3072" in warning for warning in config.plan()["async_capacity"]["warnings"]
    )
    assert run_spec(tmp_path).compile().miles["async_max_concurrent_samples"] == 6144


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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    inventory = gpu_hardware.inventory()
    assert [device["id"] for device in inventory] == ["0", "2"]
    assert inventory[0]["memory_mib"] is None
    report = profiles.resolve("olmo35-h100-v1", "ai2/jupiter", model(), options())
    with pytest.raises(ValueError, match="expects H100"):
        profiles.validate_hardware(report, [("NVIDIA H100", None)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-example")
    with pytest.raises(ValueError, match="MIG/UUID"):
        gpu_hardware.inventory()


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
    pending = profiles.resolve("olmo35-b300-v1", "ai2/holmes", None, options())
    layout = {"replicas": 3, "nodes": [{"rollout_gpus": 0}, {"rollout_gpus": 1}, {"rollout_gpus": 1}]}
    for rank, devices in enumerate([[], [["NVIDIA B300", 275040]], [["NVIDIA H100", 81559]]]):
        (tmp_path / f"inference-profile-{rank}.json").write_text(
            json.dumps({"profile": pending, "rollout_gpus": len(devices), "hardware": devices})
        )
    resolved = profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options())
    with pytest.raises(ValueError, match="^Inference profile .*expects B300"):
        profiles.check_cluster_hardware(resolved, tmp_path, layout)
    (tmp_path / "inference-profile-2.json").write_text(
        json.dumps({"rollout_gpus": 1, "hardware": [["NVIDIA B300", 275040]]})
    )
    assert len(profiles.check_cluster_hardware(resolved, tmp_path, layout)) == 3
    (tmp_path / "inference-profile-1.json").unlink()
    with pytest.raises(ValueError, match="inference-profile-1"):
        profiles.check_cluster_hardware(resolved, tmp_path, layout)


@pytest.mark.parametrize("failure", [None, "wrong_gpu", "missing_metadata"])
def test_workflow_validates_remote_inventory_after_preparation(tmp_path, monkeypatch, failure):
    run = run_spec(tmp_path, ["inference.gpus=14"])
    (tmp_path / "model" / "config.json").unlink()
    assert run.compile().inference_profile["model_metadata_available"] is False
    prepared = tmp_path / "prepared-hf"
    prepared.mkdir()
    if failure != "missing_metadata":
        (prepared / "config.json").write_text(json.dumps(model()))
    reports = tmp_path / "replicas"
    reports.mkdir()
    for rank, count in enumerate([6, 8]):
        name = "NVIDIA H100" if failure == "wrong_gpu" and rank == 1 else "NVIDIA B300"
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
    if failure:
        message = (
            "^Inference profile .*expects B300"
            if failure == "wrong_gpu"
            else "prepared checkpoint config.json is unavailable"
        )
        with pytest.raises(ValueError, match=message):
            workflow.execute(run)
        assert not trained
    else:
        assert workflow.execute(run)["status"] == "complete"
        assert len(trained) == 1
        assert trained[0].inference_profile["model_metadata_available"] is True


def test_single_node_hardware_check_only_uses_rollout_slice(monkeypatch):
    report = profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options())
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


def test_duplicate_startup_gpu_mask_is_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    with pytest.raises(ValueError, match="duplicate"):
        gpu_hardware.inventory()


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


@pytest.mark.parametrize(
    "cluster,gpu,kv,slots,fraction,cache_off",
    [("ai2/holmes", "b300", 17825792, 3072, 0.7, False), ("ai2/jupiter", "h100", 6684672, 512, 0.9, True)],
)
def test_presets_apply_verbatim(tmp_path, cluster, gpu, kv, slots, fraction, cache_off):
    for selector in (f"olmo35-{gpu}-v1",):
        config = run_spec(tmp_path, [f'launch.cluster="{cluster}"'], profile=selector).compile()
        for key, value in profiles.PRESETS[f"olmo35-{gpu}-v1"].items():
            assert config.miles[key] == value
        assert config.miles["sglang_max_running_requests"] == 512
        assert config.miles["sglang_max_total_tokens"] == kv
        assert config.miles["sglang_max_mamba_cache_size"] == slots
        assert config.miles["sglang_mem_fraction_static"] == fraction
        assert config.miles["sglang_disable_radix_cache"] is cache_off
        assert config.miles["async_max_concurrent_samples"] == 6144
        assert config.core.max_policy_lag == 6


@pytest.mark.parametrize("selector", ["olmo35-b300-v1"])
@pytest.mark.parametrize("key", profiles.PROFILE_KEYS)
def test_every_explicit_preset_key_conflicts_even_when_equal(selector, key):
    value = profiles.PRESETS["olmo35-b300-v1"][key]
    with pytest.raises(ValueError, match="Explicit preset-owned settings conflict") as raised:
        profiles.resolve(selector, "ai2/holmes", model(), options() | {key: value})
    message = str(raised.value)
    assert key in message and "Either remove" in message and "or select generic" in message
    block = tomllib.loads(message[message.index("[inference]") :])["inference"]
    assert block.pop("profile") == "generic"
    assert block == profiles.PRESETS["olmo35-b300-v1"]


@pytest.mark.parametrize(
    "override",
    [
        "inference.sglang_max_running_requests=256",
        "inference.sglang_mem_fraction_static=0.4",
        "inference.radix_cache=true",  # equal to B300 preset, still owned by it
        "miles.sglang_max_mamba_cache_size=3072",
        'inference.mamba_radix_cache_strategy="extra_buffer"',
    ],
)
def test_native_and_friendly_aliases_cannot_override_presets(tmp_path, override):
    with pytest.raises(ValueError, match="preset-owned settings conflict"):
        run_spec(tmp_path, [override])


def test_pending_metadata_conflict_still_offers_a_parseable_exit(tmp_path):
    run = run_spec(tmp_path)
    (tmp_path / "model" / "config.json").unlink()
    run.sections["inference"]["sglang_mem_fraction_static"] = 0.4
    with pytest.raises(ValueError, match="preset-owned settings conflict"):
        run.compile()


@pytest.mark.parametrize("cluster", ["ai2/holmes", "ai2/jupiter"])
@pytest.mark.parametrize("selector", [None, "generic"])
def test_legacy_configs_keep_explicit_settings_and_get_one_hint(tmp_path, cluster, selector):
    config = run_spec(
        tmp_path,
        [
            f'launch.cluster="{cluster}"',
            "inference.sglang_max_mamba_cache_size=1024",
            "inference.sglang_mem_fraction_static=0.7",
        ],
        profile=selector,
    ).compile()
    assert config.inference_profile["selected"] == "generic"
    assert config.inference_profile["defaults"] == {}
    assert config.miles["sglang_max_running_requests"] == 64
    assert config.miles["sglang_max_mamba_cache_size"] == 1024
    assert config.miles["sglang_mem_fraction_static"] == 0.7
    assert config.miles["sglang_disable_radix_cache"] is True
    assert config.miles["async_max_concurrent_samples"] == 768
    warnings = config.inference_profile["tips"]
    assert len(warnings) == 1 and warnings[0]["code"] == "inference_profile_available"
    assert "unchanged" in warnings[0]["message"] and "\n" not in warnings[0]["message"]


def test_generic_legacy_values_survive_late_metadata(tmp_path):
    run = run_spec(
        tmp_path,
        ["inference.sglang_max_mamba_cache_size=1024", "inference.sglang_mem_fraction_static=0.7"],
        profile=None,
    )
    (tmp_path / "model" / "config.json").unlink()
    before = run.compile()
    assert "inference_metadata_unavailable" in warning_map(before)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "config.json").write_text(json.dumps(model()))
    after = run.compile({"hf_checkpoint": str(prepared)})
    for key in profiles.PROFILE_KEYS:
        assert before.miles.get(key) == after.miles.get(key)
    assert "inference_profile_available" in warning_map(after)


def test_generic_malformed_metadata_does_not_block_old_configs(tmp_path):
    run = run_spec(tmp_path, profile=None)
    (tmp_path / "model" / "config.json").write_text("invalid json")
    assert "inference_metadata_unavailable" in warning_map(run.compile())


@pytest.mark.parametrize("gpu", ["b300", "h100"])
def test_generic_paste_and_tweak_is_explicit_not_derived(tmp_path, gpu):
    block = tomllib.loads(profiles.editable_toml(f"olmo35-{gpu}-v1"))["inference"]
    block["sglang_max_running_requests"] = 256
    overrides = [f"inference.{key}={json.dumps(value)}" for key, value in block.items()]
    config = run_spec(tmp_path, overrides).compile()
    assert config.miles["sglang_max_running_requests"] == 256
    assert config.miles["sglang_cuda_graph_max_bs_decode"] == 512
    assert (
        config.miles["sglang_max_mamba_cache_size"]
        == profiles.PRESETS[f"olmo35-{gpu}-v1"]["sglang_max_mamba_cache_size"]
    )
    assert config.inference_profile["defaults"] == {}


def test_auto_removed_and_named_no_match(tmp_path):
    with pytest.raises(ValueError, match="auto was removed"):
        run_spec(tmp_path, profile="auto")
    with pytest.raises(ValueError, match="dedicated TP1"):
        run_spec(tmp_path, ["inference.rollout_tensor_parallel_size=2"])
    deferred = run_spec(tmp_path)
    (tmp_path / "model" / "config.json").unlink()
    config = deferred.compile()
    assert "inference_model_guard_deferred" in warning_map(config)
    assert config.miles["sglang_max_running_requests"] == 512


def test_commands_emit_same_warning_set_and_json_is_clean(tmp_path, monkeypatch, capsys):
    run = run_spec(tmp_path, ["async.async_max_concurrent_samples=8"], profile="generic")
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
            assert plan["inference_profile"]["selected"] == "generic"
            assert captured.out.count('"code": "inference_profile_available"') == 1
    assert messages[0] == messages[1] == messages[2]
    assert "Producer budget 8" in messages[0]
    assert "Configuration tips (settings unchanged)" in messages[0]
    assert "olmo35-b300-v1" in messages[0]


def test_measured_http_equality_does_not_ask_for_different_preset(tmp_path):
    config = run_spec(tmp_path).compile()
    assert not any("Producer budget equals one wave" in w for w in config.plan()["async_capacity"]["warnings"])
    limited = run_spec(tmp_path, ["async.async_max_concurrent_samples=3072"]).compile()
    assert any("Producer budget equals one wave" in w for w in limited.plan()["async_capacity"]["warnings"])


def test_pasted_generic_preset_has_same_numeric_http_advice(tmp_path):
    overrides = [f"inference.{key}={json.dumps(value)}" for key, value in profiles.PRESETS["olmo35-b300-v1"].items()]
    config = run_spec(tmp_path, overrides, profile="generic").compile()
    assert not any("Producer budget equals one wave" in w for w in config.plan()["async_capacity"]["warnings"])


def test_advisor_flags_copied_pools_without_changing_settings(tmp_path):
    preset = profiles.PRESETS["olmo35-b300-v1"]
    overrides = ['launch.cluster="ai2/jupiter"'] + [
        f"inference.{key}={preset[key]}" for key in ("sglang_max_total_tokens", "sglang_max_mamba_cache_size")
    ]
    config = run_spec(tmp_path, overrides, profile="generic").compile()
    assert "inference_pool_hardware_mismatch" in warning_map(config)
    assert config.miles["sglang_max_mamba_cache_size"] == 3072
    assert config.miles["sglang_max_running_requests"] == 64


def test_core_template_is_not_checkpoint_metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(model()))
    source = {"format": "olmo_core", "source": str(tmp_path), "hf_template": str(tmp_path)}
    assert profiles.model_metadata(source, {}) == (None, None)
    assert profiles.model_metadata(source, {"hf_checkpoint": str(tmp_path)})[0] == model()


def test_malformed_inventory_name_is_contextual_input_error(tmp_path):
    report = profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options())
    (tmp_path / "inference-profile-0.json").write_text(json.dumps({"rollout_gpus": 1, "hardware": [[123, 275040]]}))
    with pytest.raises(ValueError, match="^Inference profile .*expects B300"):
        profiles.check_cluster_hardware(report, tmp_path, {"replicas": 1, "nodes": [{"rollout_gpus": 1}]})


@pytest.mark.parametrize(
    "change",
    [{"sglang_attention_backend": "flashinfer"}, {"sglang_page_size": 16}, {"sglang_disable_overlap_schedule": True}],
)
def test_advisor_does_not_recommend_a_preset_rejected_by_shared_capacity_rules(change):
    values = options() | change
    report = profiles.resolve("generic", "ai2/holmes", model(), values)
    _, tips = profiles.advise(report, "ai2/holmes", model(), values, set(change))
    assert tips == []


def test_malformed_inventory_json_keeps_file_context(tmp_path):
    report = profiles.resolve("olmo35-b300-v1", "ai2/holmes", model(), options())
    (tmp_path / "inference-profile-0.json").write_text("not JSON")
    with pytest.raises(ValueError, match="Cannot qualify rollout hardware from .*inference-profile-0.json"):
        profiles.check_cluster_hardware(report, tmp_path, {"replicas": 1, "nodes": [{"rollout_gpus": 1}]})
