"""Keep frozen-runtime diagnostics separate from training and preserve failure evidence."""

import base64
import hashlib
import importlib
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from scripts.miles import launch_frozen_core_score_profile as launch

ROOT = Path(__file__).resolve().parents[2]


def test_profile_uses_frozen_image_bounded_topology_and_private_rank_caches():
    task = launch.specification(launch.IMAGE)["tasks"][0]
    assert task["image"]["beaker"] == launch.IMAGE
    assert task["resources"]["gpuCount"] == 2
    assert task["context"] == {"priority": "urgent", "minRuntime": "30m", "autoResume": False}
    assert task["timeout"] == "60m"
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    command = task["arguments"][0]
    assert "test ! -e" in command
    assert "--nproc_per_node=2" in command
    assert "git checkout" not in command and "pip install" not in command
    assert "trap" in command and "rank[0-9].json" in command
    encoded_parts = re.findall(r"printf %s ([A-Za-z0-9+/=]+) \| base64", command)
    worker_text, embedded_manifest, rank_script = [base64.b64decode(value).decode() for value in encoded_parts]
    assert json.loads(embedded_manifest)["worker_sha256"] == hashlib.sha256(worker_text.encode()).hexdigest()
    assert "rank${RANK}/triton" in rank_script
    assert "rank${RANK}/inductor" in rank_script
    manifest = json.loads((ROOT / "scripts/miles/frozen_core_score_manifest.json").read_text())
    assert manifest["image"] == launch.IMAGE
    assert set(manifest["module_sha256"]) >= {"open_instruct.miles.actor", "olmo_core.kernels.swiglu"}


@pytest.mark.parametrize("kwargs", [{"image": "alias"}, {"output": launch.ROOT + "/old"}, {"rollout": 100}])
def test_profile_rejects_unfrozen_or_ambiguous_target(kwargs):
    with pytest.raises(ValueError):
        launch.specification(**{"image": launch.IMAGE, **kwargs})


def fake_executable(directory, name, text):
    path = directory / name
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(0o755)


@pytest.mark.parametrize("dirty,image,message", [(True, launch.IMAGE, "Commit"), (False, "user/alias", "immutable")])
def test_existing_image_wrapper_retains_dirty_and_alias_rejections(tmp_path, dirty, image, message):
    fake_executable(tmp_path, "git", "echo ' M file'\n" if dirty else "exit 0\n")
    fake_executable(tmp_path, "beaker", 'touch "$CALLED"\nexit 99\n')
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + ":" + os.environ["PATH"],
        MILES_EXISTING_IMAGE=image,
        CALLED=str(tmp_path / "called"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/miles/build_and_launch.sh"), "unused"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and message in result.stdout
    assert not (tmp_path / "called").exists()


def test_existing_image_wrapper_verifies_id_and_forwards_arguments(tmp_path):
    fake_executable(tmp_path, "git", "exit 0\n")
    fake_executable(tmp_path, "beaker", 'printf \'[{"id":"%s"}]\' "$MILES_EXISTING_IMAGE"\n')
    script = tmp_path / "consumer.sh"
    script.write_text('printf "%s\\n" "$@"\n')
    environment = dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"], MILES_EXISTING_IMAGE=launch.IMAGE)
    environment.pop("MILES_BASE_IMAGE", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/miles/build_and_launch.sh"), str(script), "--render-only"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [launch.IMAGE, "--render-only"]


@pytest.fixture
def worker():
    pytest.importorskip("torch")
    pytest.importorskip("miles")
    return importlib.import_module("scripts.miles.profile_frozen_core_scores")


def test_score_residual_reports_nonfinite_and_changed_outputs(worker):
    x = worker.torch.tensor([1.0, 2.0])
    assert worker.compare_scores([x], [x.clone()])["valid"]
    changed = worker.compare_scores([x], [x + 0.25])
    assert not changed["valid"] and changed["max_abs"] == 0.25
    assert not worker.compare_scores([x], [x[:1]])["valid"]
    assert not worker.compare_scores([x], [x * float("nan")])["finite"]


def test_jit_misses_and_artifact_writes_remain_distinct(worker):
    observation = worker.CompilerObservation()
    function = SimpleNamespace(fn=SimpleNamespace(__module__="fixture", __name__="kernel"))
    observation.original_compile = mock.Mock(return_value="disk-cached kernel")
    observation.original_put = mock.Mock(return_value="path")
    assert observation.compile(function, "key", {}, 0, {"rows": 19}, None, None, False) == "disk-cached kernel"
    assert observation.summary()["jit_miss_count"] == 1
    assert observation.summary()["cache_artifact_writes_by_extension"] == {}
    observation.put(None, "bytes", "kernel.cubin")
    assert observation.summary()["cache_artifact_writes_by_extension"] == {".cubin": 1}
    assert "rows" in observation.summary()["jit_in_memory_misses"][0]["constexprs"]
