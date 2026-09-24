"""Immutable generation, relocation, corruption, and real child lifecycle checks."""

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import compiler_cache_run as wrapper

from open_instruct.miles import compiler_cache as cache

KEY = "a" * 64
IMAGE = "sha256:" + "b" * 64


def private(tmp_path, name):
    root = tmp_path / name
    cache.local_environment(root)
    return root


def write(root, name, value=b"compiled", executable=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(0o700 if executable else 0o600)
    return path


def identity():
    return {
        "image": IMAGE,
        "runtime_lock": {"core": "revision"},
        "sources": {name: {"sha256": KEY} for name in ("olmo-core", "miles", "open-instruct", "olmo-sglang")},
        "model_config": {"hidden_size": 128, "per_head_qk_norm": True},
        "run_config": {"core": {"ep_size": 1}, "miles": {"context_length": 512, "save": "run-a"}},
        "toolchain": {"driver": "580", "gpu": "B300", "compiler": "13", "torch": "2.13"},
        "compiler_env": {"OLMO_USE_TORCH_GROUPED_MM": "0"},
    }


def test_fingerprint_tracks_compile_inputs_but_not_weights_outputs():
    args = identity()
    baseline = cache.fingerprint(**args)[0]
    args["run_config"]["miles"].update(save="run-b", hf_checkpoint="different-weights", wandb_group="different")
    args["run_config"]["core"]["reward_config"] = "/another/reward.json"
    assert cache.fingerprint(**args)[0] == baseline
    changes = [
        ("image", None, "sha256:" + "c" * 64),
        ("runtime_lock", "core", "new-revision"),
        ("sources", "olmo-core", {"sha256": "c" * 64}),
        ("model_config", "hidden_size", 256),
        ("toolchain", "driver", "590"),
        ("toolchain", "gpu", "4090"),
        ("toolchain", "compiler", "14"),
        ("compiler_env", "OLMO_USE_TORCH_GROUPED_MM", "1"),
    ]
    for group, key, value in changes:
        changed = copy.deepcopy(args)
        if key is None:
            changed[group] = value
        else:
            changed[group][key] = value
        assert cache.fingerprint(**changed)[0] != baseline
    for field, value in (("ep_size", 2), ("activation_checkpointing", "full")):
        changed = copy.deepcopy(args)
        changed["run_config"]["core"][field] = value
        assert cache.fingerprint(**changed)[0] != baseline
    args["run_config"]["core"]["model_config"] = "/external/config.py"
    with pytest.raises(ValueError, match="External Core"):
        cache.fingerprint(**args)


@pytest.mark.parametrize(
    "group,changes",
    [
        ("miles", {"rollout_sample_rate": 0.1}),
        ("miles", {"rollout_sample_rate": 1.0}),
        ("core", {"checkpoint_keep_last": 2, "checkpoint_keep_every": 20}),
        (
            "core",
            {
                "checkpoint_thread_count": 4,
                "checkpoint_process_count": 2,
                "checkpoint_profile": True,
                "checkpoint_compact_storage": False,
                "checkpoint_dedup_save_to_lowest_rank": True,
                "checkpoint_constant_memory_planning": False,
            },
        ),
        (
            "core",
            {
                "diagnostic_interval": 5,
                "pipeline_observation_interval": 2,
                "replay_diagnostics": True,
                "max_train_rollout_logprob_abs_diff": 0.01,
            },
        ),
        ("core", {"engine_drain_timeout": 900, "engine_update_timeout": 300, "refresh_request_timeout": 3600}),
        ("core", {"scoring_check_interval": 10, "scoring_check_tolerance": 0.01}),
        (
            "core",
            {
                "records_root": "/new/records",
                "records_responses": "sample",
                "records_response_sample_rate": 0.1,
                "selection_table": "/new/table",
                "selection_sha256": "new",
            },
        ),
        (
            "core",
            {
                "compiler_cache_diagnostics": True,
                "compiler_cache_root": "/new/tmp-7d",
                "compiler_cache_max_storage_bytes": 1024,
                "compiler_cache_publish_interval_seconds": 60,
            },
        ),
        (
            "miles",
            {
                "sglang_port": 12345,
                "sglang_host": "127.0.0.1",
                "sglang_watchdog_timeout": 900,
                "sglang_dist_timeout": 600,
                "sglang_download_dir": "/new/downloads",
            },
        ),
        (
            "miles",
            {
                "sglang_log_level": "debug",
                "sglang_log_requests": True,
                "sglang_enable_metrics": True,
                "sglang_collect_traces": True,
                "sglang_otlp_traces_endpoint": "http://localhost:4317",
            },
        ),
    ],
)
def test_operational_changes_reuse_published_cache(tmp_path, group, changes):
    args = identity()
    key = cache.fingerprint(**args)[0]
    source = private(tmp_path, "compiled")
    write(source / "triton", "kernel.so")
    cache.publish(tmp_path / "shared", source, key, "triton")
    args["run_config"][group].update(changes)
    new_key = cache.fingerprint(**args)[0]
    assert new_key == key
    restored = private(tmp_path, "restored")
    assert cache.restore(tmp_path / "shared", restored, new_key, "triton")["status"] == "hit"


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("core", "expert_parallel_size", 4),
        ("core", "row_specialization", "dynamic"),
        ("core", "packing_max_tokens", 32768),
        ("core", "compile_model", True),
        ("core", "attention_backend", "flash_4"),
        ("core", "scoring_pass_required", True),
        ("core", "router_aux_loss_weight", 0.1),
        ("core", "future_kernel_option", True),
        ("miles", "sglang_attention_backend", "triton"),
        ("miles", "sglang_dtype", "bfloat16"),
        ("miles", "sglang_cuda_graph_max_bs_decode", 64),
        ("miles", "sglang_future_kernel_option", True),
    ],
)
def test_compile_and_unknown_options_still_invalidate(group, key, value):
    args = identity()
    before = cache.fingerprint(**args)[0]
    args["run_config"][group][key] = value
    assert cache.fingerprint(**args)[0] != before


def test_compiler_environment_ignores_ephemeral_directories_but_keeps_options():
    env = {
        "SGLANG_DG_CACHE_DIR": "/tmp/engine-a",
        "TILELANG_TMP_DIR": "/tmp/tile-a",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
        "TILELANG_DISABLE_CACHE": "0",
    }
    first = wrapper.compiler_environment(env)
    env.update(SGLANG_DG_CACHE_DIR="/tmp/engine-b", TILELANG_TMP_DIR="/tmp/tile-b")
    assert wrapper.compiler_environment(env) == first
    env["SGLANG_DG_CACHE_DIR_PER_PROCESS"] = "0"
    assert wrapper.compiler_environment(env) != first


def test_source_identity_hashes_dirty_code_and_rejects_symlink(tmp_path):
    source = write(tmp_path, "module.py", b"value = 1\n")
    first = cache.source_identity(tmp_path)
    source.write_text("value = 2\n")
    assert cache.source_identity(tmp_path) != first
    (tmp_path / "linked.py").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        cache.source_identity(tmp_path)


def test_shared_hot_cache_and_unexpired_weka_root_rejected():
    with pytest.raises(ValueError, match="node-local"):
        cache.local_environment(Path("/weka/shared/hot-cache"))
    with pytest.raises(ValueError, match="expiry"):
        cache.artifact_root(Path("/weka/shared/cache"), KEY, "triton")
    assert cache.artifact_root(Path("/weka/shared/tmp-30d/cache"), KEY, "triton").name == "triton"


def test_immutable_generations_and_fresh_restore(tmp_path):
    shared = tmp_path / "shared"
    source = private(tmp_path, "first")
    original = write(source / "triton", "kernel/object.so", executable=True)
    first = cache.publish(shared, source, KEY, "triton")
    base = cache.artifact_root(shared, KEY, "triton")
    generation = base / "generations" / first["generation"]
    frozen = {p.name: p.read_bytes() for p in generation.iterdir()}
    assert cache.publish(shared, source, KEY, "triton")["status"] == "unchanged"
    write(source / "triton", "second/object.so", b"another")
    second = cache.publish(shared, source, KEY, "triton")
    assert second["generation"] != first["generation"]
    assert {p.name: p.read_bytes() for p in generation.iterdir()} == frozen
    destination = private(tmp_path, "second")
    restored = cache.restore(shared, destination, KEY, "triton")
    assert restored["status"] == "hit"
    assert cache.inventory(source / "triton") == cache.inventory(destination / "triton")
    assert os.access(destination / "triton/kernel/object.so", os.X_OK)
    assert original.read_bytes() == b"compiled"
    with pytest.raises(ValueError, match="fresh"):
        cache.restore(shared, destination, KEY, "triton")
    assert cache.restore(shared, private(tmp_path, "other-key"), "b" * 64, "triton")["status"] == "miss"


def test_concurrent_disjoint_publications_merge_and_conflicts_fail(tmp_path):
    shared = tmp_path / "shared"
    roots = [private(tmp_path, f"writer-{index}") for index in range(2)]
    for index, root in enumerate(roots):
        write(root / "tilelang", f"kernel-{index}.so", str(index).encode())
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda root: cache.publish(shared, root, KEY, "tilelang"), roots))
    assert all(report["status"] == "published" for report in reports)
    destination = private(tmp_path, "restored")
    assert cache.restore(shared, destination, KEY, "tilelang")["status"] == "hit"
    assert set(cache.inventory(destination / "tilelang")) == {"kernel-0.so", "kernel-1.so"}
    base = cache.artifact_root(shared, KEY, "tilelang")
    before = (base / "CURRENT").read_bytes()
    write(roots[0] / "tilelang", "kernel-0.so", b"conflicting bytes")
    with pytest.raises(ValueError, match="Conflicting"):
        cache.publish(shared, roots[0], KEY, "tilelang")
    assert (base / "CURRENT").read_bytes() == before


def test_triton_real_group_survives_private_root_removal(tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    runtime = pytest.importorskip("triton.runtime.cache")
    source = private(tmp_path, "original")
    monkeypatch.setattr(triton.knobs.cache, "dir", str(source / "triton"))
    manager = runtime.FileCacheManager("kernel-key")
    artifact = manager.put(b"compiled", "kernel.cubin")
    manager.put_group("kernel.json", {"kernel.cubin": artifact})
    report = cache.publish(tmp_path / "shared", source, KEY, "triton")
    assert report["status"] == "published"
    shutil.rmtree(source)
    target = private(tmp_path, "new-private")
    restored = cache.restore(tmp_path / "shared", target, KEY, "triton")
    assert restored["status"] == "hit" and restored["relocated_groups"] == 1
    monkeypatch.setattr(triton.knobs.cache, "dir", str(target / "triton"))
    group = runtime.FileCacheManager("kernel-key").get_group("kernel.json")
    assert group == {"kernel.cubin": str(target / "triton/kernel-key/kernel.cubin")}
    assert Path(group["kernel.cubin"]).read_bytes() == b"compiled"
    assert cache.publish(tmp_path / "shared", target, KEY, "triton")["generation"] == report["generation"]


@pytest.mark.parametrize("corruption", ["archive", "fingerprint", "manifest", "symlink"])
def test_corrupt_restore_rejected_without_partial_cache(tmp_path, corruption):
    source = private(tmp_path, "source")
    shared = tmp_path / "shared"
    write(source / "fa4", "artifact.so")
    result = cache.publish(shared, source, KEY, "fa4")
    generation = cache.artifact_root(shared, KEY, "fa4") / "generations" / result["generation"]
    if corruption == "archive":
        (generation / "cache.tar.gz").write_bytes(b"broken")
    elif corruption == "symlink":
        archive = generation / "cache.tar.gz"
        archive.rename(generation / "real.tar.gz")
        archive.symlink_to("real.tar.gz")
    else:
        path = generation / "manifest.json"
        data = json.loads(path.read_text())
        if corruption == "fingerprint":
            data["fingerprint"] = "b" * 64
        else:
            data["files"] = []
        path.write_bytes(cache.encoded(data))
    target = private(tmp_path, "destination")
    assert cache.restore(shared, target, KEY, "fa4")["status"] == "rejected"
    assert list((target / "fa4").iterdir()) == []
    with pytest.raises(ValueError):
        cache.publish(shared, source, KEY, "fa4")


@pytest.mark.parametrize("member_kind", ["traversal", "symlink", "duplicate"])
def test_verified_checksum_does_not_authorize_unsafe_tar_members(tmp_path, member_kind):
    shared = tmp_path / "shared"
    source = private(tmp_path, "source")
    write(source / "deepep", "artifact.so")
    result = cache.publish(shared, source, KEY, "deepep")
    generation = cache.artifact_root(shared, KEY, "deepep") / "generations" / result["generation"]
    archive = generation / "cache.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        entry = tarfile.TarInfo("../outside" if member_kind == "traversal" else "artifact.so")
        entry.size = len(b"compiled")
        if member_kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/tmp/outside"
        output.addfile(entry, io.BytesIO(b"compiled"))
        if member_kind == "duplicate":
            output.addfile(entry, io.BytesIO(b"compiled"))
    path = generation / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["archive_sha256"] = cache.sha256(archive)
    path.write_bytes(cache.encoded(manifest))
    target = private(tmp_path, "destination")
    assert cache.restore(shared, target, KEY, "deepep")["status"] == "rejected"
    assert list((target / "deepep").iterdir()) == []


def test_unreferenced_corrupt_generation_is_not_promoted(tmp_path):
    shared = tmp_path / "shared"
    source = private(tmp_path, "source")
    write(source / "deepgemm", "artifact.so")
    result = cache.publish(shared, source, KEY, "deepgemm")
    base = cache.artifact_root(shared, KEY, "deepgemm")
    (base / "CURRENT").unlink()
    (base / "generations" / result["generation"] / "cache.tar.gz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        cache.publish(shared, source, KEY, "deepgemm")
    assert not (base / "CURRENT").exists()


def test_unknown_embedded_private_path_and_links_rejected(tmp_path):
    source = private(tmp_path, "source")
    write(source / "torchinductor", "metadata", str(source / "torchinductor/artifact.so").encode())
    with pytest.raises(ValueError, match="embedded"):
        cache.publish(tmp_path / "shared", source, KEY, "torchinductor")
    (source / "triton/link").symlink_to("/tmp/unknown")
    with pytest.raises(ValueError, match="link"):
        cache.publish(tmp_path / "shared", source, KEY, "triton")


def options(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "toolchain", lambda environment: {"test_only_gpu": "fake"})
    runtime_lock = write(tmp_path, "lock.json", b'{"revision":"fixed"}')
    hf_config = write(tmp_path, "hf-config.json", b'{"hidden_size":128}')
    run_config = write(tmp_path, "run.toml", b"[core]\nep_size=1\n[miles]\nnum_rollout=1\n")
    source = []
    for name in ("olmo-core", "miles", "open-instruct", "olmo-sglang"):
        root = tmp_path / "sources" / name
        write(root, "module.py", b"value = 1\n")
        source.append(f"{name}={root}")
    return SimpleNamespace(
        report=tmp_path / "cold.json",
        command=[],
        local_parent=tmp_path / "local",
        mode="cold",
        publish=True,
        shared_root=tmp_path / "shared",
        image=IMAGE,
        runtime_lock=runtime_lock,
        hf_config=hf_config,
        run_config=run_config,
        source=source,
    )


def test_real_child_cold_publish_restored_run_and_timings(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    args.command = [
        sys.executable,
        "-c",
        "import os,pathlib; pathlib.Path(os.environ['TRITON_CACHE_DIR'],'kernel.so').write_bytes(b'compiled')",
    ]
    assert wrapper.run(args) == 0
    cold = json.loads(args.report.read_text())
    assert cold["status"] == "completed"
    assert cold["publish"][0]["status"] == "published"
    assert not Path(cold["private_local_root"]).exists()
    args.mode = "restore"
    args.report = tmp_path / "restored.json"
    args.command = [
        sys.executable,
        "-c",
        "import os,pathlib; assert pathlib.Path(os.environ['TRITON_CACHE_DIR'],'kernel.so').read_bytes()==b'compiled'",
    ]
    assert wrapper.run(args) == 0
    restored = json.loads(args.report.read_text())
    assert restored["restore"][0]["status"] == "hit"
    assert restored["fingerprint"] == cold["fingerprint"]
    assert restored["private_local_root"] != cold["private_local_root"]
    assert not Path(restored["private_local_root"]).exists()
    for result in (cold, restored):
        assert result["total_seconds"] >= result["command_seconds"] > 0
        assert result["probe_seconds"] > 0
    with pytest.raises(ValueError, match="new report"):
        wrapper.run(args)


def test_failed_child_never_publishes(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    args.command = [sys.executable, "-c", "raise SystemExit(7)"]
    assert wrapper.run(args) == 7
    report = json.loads(args.report.read_text())
    assert report["status"] == "child_failed" and report["publish"] == []
    assert not args.shared_root.exists()


def test_surviving_child_group_stops_publication_and_retains_evidence(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    args.command = [
        sys.executable,
        "-c",
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])",
    ]
    with pytest.raises(RuntimeError, match="process group"):
        wrapper.run(args)
    report = json.loads(args.report.read_text())
    assert report["status"] == "wrapper_failed"
    assert report["publish"] == [] and report["private_cache_retained"]
    assert Path(report["private_local_root"]).is_dir()


def test_compiler_environment_excludes_private_paths_but_tracks_precision():
    env = {
        "TRITON_CACHE_DIR": "/tmp/private",
        "CUDA_VISIBLE_DEVICES": "0",
        "OLMO_USE_TORCH_GROUPED_MM": "0",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    assert wrapper.compiler_environment(env) == {
        "NVIDIA_TF32_OVERRIDE": cache.digest("0"),
        "OLMO_USE_TORCH_GROUPED_MM": cache.digest("0"),
    }


def test_changed_runtime_source_blocks_publication(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    source = tmp_path / "sources/olmo-core/module.py"
    args.command = [
        sys.executable,
        "-c",
        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('changed = True')",
        str(source),
    ]
    with pytest.raises(ValueError, match="changed while"):
        wrapper.run(args)
    report = json.loads(args.report.read_text())
    assert report["status"] == "wrapper_failed" and report["publish"] == []
    assert report["child_returncode"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "OLMO_HF_MOE_CORE_REFERENCE",
        "NVTE_FUSED_ATTN",
        "FLA_USE_FAST_OPS",
        "TILELANG_TARGET",
        "FLASH_ATTENTION_VARIANT",
        "CUTE_BACKEND",
        "CUDNN_ENABLED",
    ],
)
def test_compile_environment_changes_are_fingerprinted_without_raw_values(name):
    first = wrapper.compiler_environment({name: "0", "SGLANG_API_KEY": "private-credential"})
    second = wrapper.compiler_environment({name: "1", "SGLANG_API_KEY": "private-credential"})
    assert first != second
    assert "private-credential" not in json.dumps(first)


@pytest.mark.parametrize(
    "name",
    [
        "TRITON_CACHE_MANAGER",
        "TRITON_REMOTE_CACHE_BACKEND",
        "TRITON_OVERRIDE_DIR",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    ],
)
def test_remote_or_custom_cache_controls_rejected(name):
    with pytest.raises(ValueError, match="not qualified"):
        wrapper.validate_local_cache_controls({name: "1"})
    wrapper.validate_local_cache_controls({"TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0"})


def test_interrupted_wait_stops_owned_child_and_records_evidence(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    processes = []

    class InterruptOnce(subprocess.Popen):
        interrupted = False

        def wait(self, *arguments, **kwargs):
            if not self.interrupted:
                self.interrupted = True
                processes.append(self)
                raise KeyboardInterrupt("simulated wrapper signal")
            return super().wait(*arguments, **kwargs)

    monkeypatch.setattr(wrapper.subprocess, "Popen", InterruptOnce)
    args.command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with pytest.raises(KeyboardInterrupt):
        wrapper.run(args)
    assert processes[0].poll() is not None
    report = json.loads(args.report.read_text())
    assert report["status"] == "wrapper_failed" and report["publish"] == []
    assert report["private_cache_retained"]


def test_corrupt_cache_is_cold_fallback_but_never_repaired_implicitly(tmp_path, monkeypatch):
    args = options(tmp_path, monkeypatch)
    args.command = [
        sys.executable,
        "-c",
        "import os,pathlib;pathlib.Path(os.environ['TRITON_CACHE_DIR'],'kernel.so').write_bytes(b'compiled')",
    ]
    assert wrapper.run(args) == 0
    cold = json.loads(args.report.read_text())
    generation = (
        cache.artifact_root(args.shared_root, cold["fingerprint"], "triton")
        / "generations"
        / cold["publish"][0]["generation"]
    )
    archive = generation / "cache.tar.gz"
    archive.write_bytes(b"corrupt")
    args.mode = "restore"
    args.report = tmp_path / "after-corruption.json"
    assert wrapper.run(args) == 0
    report = json.loads(args.report.read_text())
    assert report["restore"][0]["status"] == "rejected"
    assert report["publish"][0]["status"] == "rejected"
    assert report["status"] == "completed" and archive.read_bytes() == b"corrupt"


def test_publication_stages_tree_locally_and_upload_failure_keeps_current(tmp_path, monkeypatch):
    local = private(tmp_path, "local")
    shared = tmp_path / "shared"
    write(local, "triton/kernel", b"first")
    first = cache.publish(shared, local, KEY, "triton")
    write(local, "triton/new", b"second")
    original_snapshot = cache.snapshot
    original_copy = cache.shutil.copy2

    def snapshot(source, destination, family):
        assert destination.is_relative_to(local)
        return original_snapshot(source, destination, family)

    def failed_upload(source, destination, *args, **kwargs):
        if Path(destination).is_relative_to(shared):
            assert Path(destination).name in {"cache.tar.gz", "manifest.json"}
            if Path(destination).name == "manifest.json":
                raise OSError("injected upload failure")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(cache, "snapshot", snapshot)
    monkeypatch.setattr(cache.shutil, "copy2", failed_upload)
    with pytest.raises(OSError, match="injected upload"):
        cache.publish(shared, local, KEY, "triton")
    base = cache.artifact_root(shared, KEY, "triton")
    assert (base / "CURRENT").read_text().strip() == first["generation"]
    assert len(list((base / "generations").iterdir())) == 1
    restored = private(tmp_path, "restored")
    assert cache.restore(shared, restored, KEY, "triton")["status"] == "hit"
    assert (restored / "triton/kernel").read_bytes() == b"first"
    assert not (restored / "triton/new").exists()
    monkeypatch.setattr(cache.shutil, "copy2", original_copy)
    events = []
    result = cache.publish(shared, local, KEY, "triton", progress=events.append)
    assert result["files"] == 2 and result["archive_bytes"] > 0
    assert {"lock_wait", "merge", "archive_local", "upload", "pointer", "complete"} <= {e["phase"] for e in events}
    assert result["phase_seconds"]["upload"] >= 0


def test_storage_cap_counts_previous_runs_and_preserves_last_good_generation(tmp_path):
    shared = tmp_path / "shared"
    first = private(tmp_path, "first-run")
    write(first / "triton", "old.so", os.urandom(4096))
    cap = 8000
    result = cache.publish(shared, first, KEY, "triton", max_storage_bytes=cap)
    assert result["status"] == "published"
    base = cache.artifact_root(shared, KEY, "triton")
    old_pointer = (base / "CURRENT").read_bytes()
    before = cache.storage_bytes(base.parent)
    assert before < cap
    second = private(tmp_path, "second-run")
    write(second / "triton", "new.so", os.urandom(1024))
    refused = cache.publish(shared, second, KEY, "triton", max_storage_bytes=cap)
    assert refused["status"] == "storage_limit"
    assert refused["storage_bytes"] == before
    assert before + refused["required_bytes"] > cap
    assert cache.storage_bytes(base.parent) == before
    assert (base / "CURRENT").read_bytes() == old_pointer
    restored = private(tmp_path, "still-readable")
    assert cache.restore(shared, restored, KEY, "triton")["status"] == "hit"
    assert (restored / "triton" / "old.so").read_bytes() == (first / "triton" / "old.so").read_bytes()
    # Another fresh run does not acquire another allowance for the same key.
    third = private(tmp_path, "third-run")
    write(third / "triton", "another.so", os.urandom(1024))
    assert cache.publish(shared, third, KEY, "triton", max_storage_bytes=cap)["status"] == "storage_limit"
    # Distinct compile identities have independent budgets.
    assert cache.publish(shared, third, "b" * 64, "triton", max_storage_bytes=cap)["status"] == "published"


def test_concurrent_families_share_one_key_storage_budget(tmp_path):
    shared = tmp_path / "shared"
    families = ["tilelang", "fa4"]
    roots = [private(tmp_path, family) for family in families]
    for root, family in zip(roots, families, strict=True):
        write(root / family, "object.so", os.urandom(4096))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: cache.publish(shared, item[0], KEY, item[1], max_storage_bytes=8000),
                zip(roots, families, strict=True),
            )
        )
    assert sorted(r["status"] for r in results) == ["published", "storage_limit"]
    key_root = cache.artifact_root(shared, KEY, "triton").parent
    assert cache.storage_bytes(key_root) <= 8000
    assert len(list(key_root.rglob("cache.tar.gz"))) == 1


def test_cap_counts_old_generations_and_interrupted_uploads(tmp_path):
    shared = tmp_path / "shared"
    root = private(tmp_path, "local")
    write(root / "triton", "a.so", b"a")
    cache.publish(shared, root, KEY, "triton")
    write(root / "triton", "b.so", b"b")
    cache.publish(shared, root, KEY, "triton")
    key_root = cache.artifact_root(shared, KEY, "triton").parent
    assert len(list(key_root.rglob("cache.tar.gz"))) == 2
    used = cache.storage_bytes(key_root)
    write(key_root, "fa4/generations/.publish-interrupted/cache.tar.gz", b"x" * 1000)
    assert cache.storage_bytes(key_root) == used + 1000
    result = cache.publish(shared, root, KEY, "triton", max_storage_bytes=used + 1000)
    assert result["status"] == "storage_limit"
    assert result["storage_bytes"] == used + 1000


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_invalid_storage_limit_cannot_publish(tmp_path, limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        cache.publish(tmp_path / "shared", tmp_path / "local", KEY, "triton", max_storage_bytes=limit)


def test_generation_recency_updates_without_mutating_archive_or_manifest(tmp_path):
    shared = tmp_path / "shared"
    source = private(tmp_path, "source")
    write(source / "triton", "kernel.so")
    first = cache.publish(shared, source, KEY, "triton")
    assert first["recency_recorded"] is True
    base = cache.artifact_root(shared, KEY, "triton")
    generation = base / "generations" / first["generation"]
    frozen = {name: (generation / name).read_bytes() for name in ("manifest.json", "cache.tar.gz")}
    marker = generation / ".last_used"
    assert marker.stat().st_size == 0
    os.utime(marker, (1, 1))
    target = private(tmp_path, "restored")
    result = cache.restore(shared, target, KEY, "triton")
    assert result["status"] == "hit" and result["recency_recorded"] is True
    assert marker.stat().st_mtime > 1
    assert {name: (generation / name).read_bytes() for name in frozen} == frozen
    os.utime(marker, (1, 1))
    result = cache.publish(shared, source, KEY, "triton")
    assert result["status"] == "unchanged" and result["recency_recorded"] is True
    assert marker.stat().st_mtime > 1


def test_recency_write_failure_does_not_prevent_cache_hit(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    source = private(tmp_path, "source")
    write(source / "triton", "kernel.so")
    cache.publish(shared, source, KEY, "triton")

    def readonly(*args, **kwargs):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(cache.os, "utime", readonly)
    target = private(tmp_path, "restored")
    result = cache.restore(shared, target, KEY, "triton")
    assert result["status"] == "hit" and result["recency_recorded"] is False
    assert (target / "triton" / "kernel.so").is_file()
