"""Immutable-runtime and bounded-resource checks for update-zero diagnostics."""

import base64
import hashlib
import json
import shlex
import subprocess

import pytest
from scripts.miles import launch_update_zero


@pytest.mark.parametrize("backend", ["core", "megatron"])
def test_launch_preserves_images_inputs_and_bounded_placement(backend):
    image = launch_update_zero.IMAGES[backend]
    spec = launch_update_zero.specification(image, backend)
    task = spec["tasks"][0]
    assert task["image"]["beaker"] == image
    assert task["resources"]["gpuCount"] == 3
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"] == {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    assert task["timeout"] == "90m"
    command = task["arguments"][0]
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    embedded = {}
    for line in command.splitlines():
        if line.startswith("printf %s "):
            fields = shlex.split(line)
            embedded[fields[-1].rsplit("/", 1)[-1]] = base64.b64decode(fields[2])
    manifest = json.loads(embedded["manifest.json"])
    for name, digest in manifest["diagnostic_file_sha256"].items():
        assert hashlib.sha256(embedded[name]).hexdigest() == digest
    inputs = json.loads(embedded["inputs.json"])
    assert len(inputs["cases"]) == 4
    assert len({c["case_id"] for c in inputs["cases"]}) == 4
    assert all(0 < len(c["input_ids"]) <= 1024 for c in inputs["cases"])
    assert all(c["original_next_token"]["core"] != c["original_next_token"]["megatron"] for c in inputs["cases"])
    assert "runpy.run_path('/usr/lib/python3.12/sitecustomize.py')" in embedded["sitecustomize.py"].decode()
    assert "SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models" in command
    assert "test ! -e " + launch_update_zero.ROOT + "/update-zero-20260911-v1/" + backend in command
    if backend == "megatron":
        assert manifest["diagnostic_recipe"]["num_optimizer_updates"] == 0
        assert "--moe-router-dtype fp32 --moe-router-use-torch-mm" in command
        assert "--num-rollout 0" in command


@pytest.mark.parametrize("backend", ["core", "megatron"])
def test_wrong_runtime_rejected(backend):
    with pytest.raises(ValueError, match="immutable"):
        launch_update_zero.specification("latest", backend)


@pytest.mark.parametrize("campaign", ["../core", "core", "update-zero-a/b", "update-zero-$(touch x)"])
def test_output_must_be_distinct_campaign(campaign):
    with pytest.raises(ValueError, match="distinct"):
        launch_update_zero.specification(launch_update_zero.IMAGES["core"], "core", campaign=campaign)


@pytest.mark.parametrize("backend", ["core", "megatron"])
def test_hf_matched_launch_uses_one_gpu_and_fresh_explicit_caches(backend):
    spec = launch_update_zero.specification(launch_update_zero.IMAGES[backend], backend, mode="hf-matched")
    task = spec["tasks"][0]
    assert task["resources"]["gpuCount"] == 1
    command = task["arguments"][0]
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "export OI_UPDATE_ZERO_MODE=hf-matched" in command
    assert "test ! -e /tmp/zero-probe/compiler-cache" in command
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"):
        assert f"export {name}=/tmp/zero-probe/compiler-cache/" in command


def test_trainer_route_mode_embeds_qualified_probe_with_original_resources():
    spec = launch_update_zero.specification(launch_update_zero.IMAGES["core"], "core", mode="trainer-routes")
    task = spec["tasks"][0]
    assert task["resources"]["gpuCount"] == 3
    command = task["arguments"][0]
    assert "export OI_UPDATE_ZERO_MODE=trainer-routes" in command
    assert "/tmp/zero-probe/update_zero_training_capture.py" in command
    assert "export TRITON_CACHE_DIR=" not in command
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)


def test_pin_profile_is_only_allowed_for_hf_control():
    with pytest.raises(ValueError, match="independent HF control"):
        launch_update_zero.specification(launch_update_zero.IMAGES["core"], "core", pin_autotune_a=True)


def test_pin_profile_is_embedded_with_provenance():
    spec = launch_update_zero.specification(
        launch_update_zero.IMAGES["core"], "core", mode="hf-matched", pin_autotune_a=True
    )
    command = spec["tasks"][0]["arguments"][0]
    assert "export OI_UPDATE_ZERO_AUTOTUNE_REFERENCE=/tmp/zero-probe/autotune-reference.json" in command
    assert "/tmp/zero-probe/update_zero_autotune.py" in command
    embedded = {}
    for line in command.splitlines():
        if line.startswith("printf %s "):
            fields = shlex.split(line)
            embedded[fields[-1].rsplit("/", 1)[-1]] = base64.b64decode(fields[2])
    manifest = json.loads(embedded["manifest.json"])
    for name in ("update_zero_autotune.py", "autotune-reference.json"):
        assert hashlib.sha256(embedded[name]).hexdigest() == manifest["diagnostic_file_sha256"][name]
