"""Worker isolation, cold fallback and successful-teardown publication contract."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from ray._private.ray_constants import WORKER_PROCESS_SETUP_HOOK_ENV_VAR
from scripts.miles import launch_startup_trial

from open_instruct.miles import startup_cache as startup
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.timing import startup_stage


def policy(root, *, restore=True):
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    return dict(
        shared=str(root / "shared"),
        report_dir=str(report_dir),
        slot="train-actor-cell0-rank0",
        image="sha256:" + "a" * 64,
        runtime_lock={},
        sources={k: {"sha256": "same"} for k in ("olmo-core", "miles", "open-instruct", "olmo-sglang")},
        model_config={},
        run_config={"core": {}, "miles": {}},
        restore=restore,
        observe=False,
    )


def test_explicit_worker_environment_preserves_existing_flags_and_separates_roles(tmp_path):
    args = SimpleNamespace(olmo_core_startup_cache=policy(tmp_path))
    train = startup.worker_runtime_env(args, "train-rank0", {"NCCL_CUMEM_ENABLE": "1"})
    serve = startup.worker_runtime_env(args, "serve-rank0", {})
    assert train["env_vars"]["NCCL_CUMEM_ENABLE"] == "1"
    assert train["worker_process_setup_hook"].endswith(".setup_worker")
    assert train["env_vars"][WORKER_PROCESS_SETUP_HOOK_ENV_VAR] == train["worker_process_setup_hook"]
    assert json.loads(train["env_vars"][startup.ENV])["slot"] != json.loads(serve["env_vars"][startup.ENV])["slot"]
    args.olmo_core_startup_cache = None
    assert startup.worker_runtime_env(args, "unused", {"X": "1"}) == {"env_vars": {"X": "1"}}


def test_worker_restore_publish_and_rank_isolation(tmp_path, monkeypatch):
    descriptor = policy(tmp_path, restore=False)
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    fake_ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node"))
    original_import = startup.importlib.import_module
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else original_import(name)
    )
    monkeypatch.setattr(startup.probes, "toolchain", lambda env: {"gpu": "B300"})
    monkeypatch.setattr(startup.probes, "compiler_environment", lambda env: {})
    startup.setup_worker()
    cold = json.loads(next((tmp_path / "reports").glob("*.json")).read_text())
    local = Path(cold["local"])
    (local / "triton/kernel").write_bytes(b"compiled")
    assert startup.publish_worker(cold)["publish"]["status"] == "published"
    assert not local.exists()
    descriptor["restore"] = True
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    startup.setup_worker()
    reports = [json.loads(p.read_text()) for p in (tmp_path / "reports").glob("*.json")]
    warm = next(r for r in reports if r["restore"]["status"] == "hit")
    assert warm["fingerprint"] == cold["fingerprint"] and warm["local"] != cold["local"]
    assert (Path(warm["local"]) / "triton/kernel").read_bytes() == b"compiled"
    startup.publish_worker(warm)
    descriptor["slot"] = "train-actor-cell0-rank1"
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    startup.setup_worker()
    other = next(
        json.loads(p.read_text())
        for p in (tmp_path / "reports").glob("*.json")
        if json.loads(p.read_text())["slot"] == descriptor["slot"]
    )
    assert other["fingerprint"] != cold["fingerprint"] and other["restore"]["status"] == "miss"
    startup.publish_worker(other)


def test_failed_run_never_publishes(tmp_path):
    args = SimpleNamespace(olmo_core_startup_cache=policy(tmp_path), save=str(tmp_path))
    with mock.patch.object(startup, "publish_worker", side_effect=AssertionError("published failed run")):
        asyncio.run(startup.finish(args, success=False))
    assert json.loads((tmp_path / "compiler-cache.json").read_text())["success"] is False


def test_disabled_cache_needs_no_sources_or_filesystem(tmp_path):
    args = SimpleNamespace(olmo_core=CoreConfig(compiler_cache=False))
    startup.prepare(args)
    assert args.olmo_core_startup_cache is None


def test_startup_timer_records_success_and_failure(tmp_path):
    args = SimpleNamespace(save=str(tmp_path), rank=2)
    device = mock.Mock()
    with startup_stage(args, "build", device=device):
        pass
    with pytest.raises(ValueError), startup_stage(args, "restore"):
        raise ValueError("injected")
    device.synchronize.assert_called_once()
    rows = [json.loads(s) for s in (tmp_path / "startup_rank2.jsonl").read_text().splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
    assert all(r["seconds"] >= 0 for r in rows)


def test_cache_controls_validate_types():
    for key in ("compiler_cache", "compiler_cache_restore", "compiler_cache_diagnostics"):
        with pytest.raises(ValueError, match=key):
            CoreConfig(**{key: "false"})


def test_trial_resource_bounds():
    task = launch_startup_trial.specification("image")["tasks"][0]
    assert task["resources"]["gpuCount"] == 3
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == "1h"
    assert task["timeout"] == "2h"


def test_cache_is_opt_out_and_uses_olmo_miles_ttl_namespace():
    assert CoreConfig().compiler_cache is True
    assert CoreConfig(compiler_cache=False).compiler_cache is False
    assert startup.DEFAULT_SHARED == "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"
    startup.cache.validate_shared_root(Path(startup.DEFAULT_SHARED))


@pytest.mark.parametrize(
    "root", ("relative/cache", "/weka", "/weka/oe-training-default/cache", "/weka/tmp-0d/cache", "", 42)
)
def test_invalid_cache_root_rejected_before_launch(root):
    with pytest.raises(ValueError):
        CoreConfig(compiler_cache_root=root)


def test_invalid_retention_path_cannot_create_reports():
    args = SimpleNamespace(olmo_core=SimpleNamespace(compiler_cache=True, compiler_cache_root="/weka/no-ttl/cache"))
    with (
        mock.patch.object(Path, "mkdir", side_effect=AssertionError("created before validation")),
        pytest.raises(ValueError, match="expiry"),
    ):
        startup.prepare(args)
