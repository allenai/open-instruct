"""Protocol ordering without importing Ray, CUDA, or the live trainer stack."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    modules = {}
    for name in (
        "ray",
        "requests",
        "torch",
        "miles",
        "miles.ray",
        "miles.utils",
        "miles.utils.tracking_utils",
        "miles.utils.tracking_utils.tracking",
    ):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    modules["miles.ray"].placement_group = SimpleNamespace()
    modules["miles.utils"].arguments = SimpleNamespace()
    modules["miles.utils"].object_store = SimpleNamespace()
    modules["miles.utils.tracking_utils.tracking"].finish_tracking = lambda: None
    modules["miles.utils.tracking_utils.tracking"].init_tracking = lambda args: None
    path = Path(__file__).parents[2] / "scripts/miles/update_zero_driver.py"
    spec = importlib.util.spec_from_file_location("_isolated_update_zero_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Remote:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def protocol(driver, tmp_path, monkeypatch, *, fail=None, cleanup_fail=False):
    events = []
    args = SimpleNamespace(
        num_rollout=0,
        use_critic=False,
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=True,
        check_weight_update_skip_list=[],
        check_weight_update_selector="all",
    )

    def record(name, result=None):
        def call(*unused, **kwargs):
            events.append(name)
            if name == fail or (cleanup_fail and name == "learner.dispose"):
                raise RuntimeError(name)
            return result

        return call

    engine = SimpleNamespace(
        get_topology_info=Remote(record("topology", {"url": "http://engine"})),
        get_server_info=Remote(record("settings", {})),
    )
    manager = SimpleNamespace(
        get_updatable_engines_and_lock=Remote(record("engines", SimpleNamespace(rollout_engines=[engine]))),
        check_weights=Remote(lambda action, **kwargs: record(action, {"valid": True})()),
        dispose=Remote(record("manager.dispose")),
    )

    class Learner:
        async def update_weights(self):
            return record("publish")()

        async def dispose(self):
            return record("learner.dispose")()

        async def train(self):
            pytest.fail("A zero-update diagnostic must never train")

    def create_rollout(*unused):
        assert args.check_weight_update_equal is False
        return record("manager.init", (manager, None))()

    async def create_training(*unused):
        assert args.check_weight_update_equal is True
        return record("learner.init", (Learner(), None))()

    driver.placement_group.create_placement_groups = record("groups", {"rollout": None})
    driver.placement_group.create_rollout_manager = create_rollout
    driver.placement_group.create_training_models = create_training
    driver.object_store.init_instance = record("store")
    driver.init_tracking = record("tracking.init")
    driver.finish_tracking = record("tracking.finish")
    driver.phase_probe = lambda url, phase, inputs, output: record(phase)()
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text("{}")
    monkeypatch.setenv("OI_UPDATE_ZERO_INPUTS", str(inputs_path))
    return args, events


def test_zero_update_protocol_order(driver, tmp_path, monkeypatch):
    args, events = protocol(driver, tmp_path, monkeypatch)
    asyncio.run(driver.probe(args, {"cases": [{}] * 4}, tmp_path))
    assert events == [
        "groups",
        "store",
        "tracking.init",
        "manager.init",
        "engines",
        "topology",
        "settings",
        "hf",
        "snapshot",
        "reset_tensors",
        "learner.init",
        "publish",
        "compare",
        "published",
        "learner.dispose",
        "manager.dispose",
        "tracking.finish",
    ]
    assert args.check_weight_update_equal is True
    assert json.loads((tmp_path / "probe-complete.json").read_text())["optimizer_calls"] == 0


@pytest.mark.parametrize("fail", ["manager.init", "hf", "snapshot", "learner.init", "publish", "compare", "published"])
def test_failure_restores_namespace_and_cleans_all_available_handles(driver, tmp_path, monkeypatch, fail):
    args, events = protocol(driver, tmp_path, monkeypatch, fail=fail, cleanup_fail=True)
    with pytest.raises(RuntimeError, match=fail):
        asyncio.run(driver.probe(args, {"cases": [{}] * 4}, tmp_path))
    assert args.check_weight_update_equal is True
    assert events[-1] == "tracking.finish"
    if fail != "manager.init":
        assert "manager.dispose" in events
    assert not (tmp_path / "probe-complete.json").exists()
    assert (tmp_path / "cleanup.json").exists()


def test_cleanup_error_does_not_skip_manager_and_is_failure(driver, tmp_path, monkeypatch):
    args, events = protocol(driver, tmp_path, monkeypatch, cleanup_fail=True)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(driver.probe(args, {"cases": [{}] * 4}, tmp_path))
    assert events[-2:] == ["manager.dispose", "tracking.finish"]
    assert not json.loads((tmp_path / "cleanup.json").read_text())["completed"]


def test_missing_capture_preserves_http_response_and_disarms(driver, tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    trace.mkdir()
    monkeypatch.setenv("OI_UPDATE_ZERO_TRACE_DIR", str(trace))
    driver.requests.post = lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"text": "retained response"}
    )
    with pytest.raises(RuntimeError, match="Expected one TP1"):
        driver.phase_probe("http://engine", "hf", {"cases": [{"case_id": "341", "input_ids": [1, 2]}]}, tmp_path)
    assert json.loads((tmp_path / "hf-341-capture-response.json").read_text())["text"] == "retained response"
    assert not (trace / "capture-request.json").exists()


def test_core_override_keeps_frozen_lr_horizon(driver, tmp_path, monkeypatch):
    config = SimpleNamespace(
        miles={"num_rollout": 100, "lr_decay_iters": 100, "use_wandb": True}, arguments=lambda: ["--frozen"]
    )
    monkeypatch.setattr(
        driver.importlib, "import_module", lambda name: SimpleNamespace(configuration=lambda root: config)
    )
    driver.arguments.parse_args = lambda: config.miles.copy()
    resolved = driver.core_arguments(tmp_path, tmp_path)
    assert resolved["num_rollout"] == 0 and resolved["lr_decay_iters"] == 100
    assert "use_wandb" not in resolved


def test_original_source_hash_gate_before_runtime(driver, tmp_path, monkeypatch):
    original = tmp_path / "original.py"
    original.write_text("unchanged source")
    checksum = driver.hashlib.sha256(original.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"original_recipe": {"module_sha256": {"original.module": checksum}}})
    )
    monkeypatch.setattr(driver.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(original)))
    assert driver.verify_original_sources(tmp_path)["original.module"]["sha256"] == checksum
    original.write_text("different source")
    with pytest.raises(ValueError, match="Original source hash mismatch"):
        driver.verify_original_sources(tmp_path)
    assert (tmp_path / "original-source-verification.json").exists()


def test_matched_hf_mode_never_initializes_or_resets_trainer(driver, tmp_path, monkeypatch):
    args, events = protocol(driver, tmp_path, monkeypatch)
    args.debug_rollout_only = False
    args.sglang_max_total_tokens = None
    args.sglang_sampling_backend = "flashinfer"
    args.sglang_mamba_radix_cache_strategy = "extra_buffer"
    monkeypatch.setenv("OI_UPDATE_ZERO_MODE", "hf-matched")
    asyncio.run(driver.probe(args, {"cases": [{}] * 4}, tmp_path))
    assert events == [
        "groups",
        "store",
        "tracking.init",
        "manager.init",
        "engines",
        "topology",
        "settings",
        "hf",
        "manager.dispose",
        "tracking.finish",
    ]
    assert args.debug_rollout_only and args.sglang_max_total_tokens == 32768
    assert args.sglang_sampling_backend == "pytorch"
    assert args.sglang_mamba_radix_cache_strategy == "auto"
    assert not (tmp_path / "probe-complete.json").exists()
    result = json.loads((tmp_path / "hf-only-complete.json").read_text())
    assert result["initial_publications"] == result["optimizer_calls"] == 0
    assert result["full_protocol_complete"] is False
    changes = json.loads((tmp_path / "serving-overrides.json").read_text())
    assert changes["sglang_sampling_backend"] == {"original": "flashinfer", "diagnostic": "pytorch"}


def test_matched_mode_rejects_unrecognized_runtime_argument(driver, tmp_path, monkeypatch):
    monkeypatch.setenv("OI_UPDATE_ZERO_MODE", "hf-matched")
    with pytest.raises(ValueError, match="lacks matched-serving"):
        driver.apply_probe_mode(SimpleNamespace(), tmp_path)


def test_unknown_mode_rejected(driver, tmp_path, monkeypatch):
    monkeypatch.setenv("OI_UPDATE_ZERO_MODE", "typo")
    with pytest.raises(ValueError, match="Unknown diagnostic mode"):
        driver.apply_probe_mode(SimpleNamespace(), tmp_path)


def test_trainer_probe_runs_after_initial_weight_comparison(driver, tmp_path, monkeypatch):
    args, events = protocol(driver, tmp_path, monkeypatch)
    monkeypatch.setenv("OI_UPDATE_ZERO_MODE", "trainer-routes")

    async def capture(*unused):
        events.append("trainer.routes")

    driver.trainer_route_probe = capture
    asyncio.run(driver.probe(args, {"cases": [{}] * 4}, tmp_path))
    assert events.index("compare") < events.index("published") < events.index("trainer.routes")
    assert events.index("trainer.routes") < events.index("learner.dispose")


def test_retained_training_payload_rejects_changed_bytes_before_loading(driver, tmp_path, monkeypatch):
    path = tmp_path / "core/rollouts/0.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong retained artifact")
    monkeypatch.setattr(driver.importlib, "import_module", lambda name: SimpleNamespace())
    driver.torch.load = lambda *args, **kwargs: pytest.fail("Changed artifact must not be loaded")
    with pytest.raises(ValueError, match="artifact hash changed"):
        driver.trainer_route_payloads(tmp_path, {})
