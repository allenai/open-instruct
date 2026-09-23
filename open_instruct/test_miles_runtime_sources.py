"""Runtime source reconstruction works for both direct pins and patched sources."""

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.miles import build_image, launch_hero_serving, prepare_runtime


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    prepare_runtime.run("git", "init", str(repo))
    prepare_runtime.run("git", "config", "user.email", "test@example.com", cwd=repo)
    prepare_runtime.run("git", "config", "user.name", "Test", cwd=repo)
    (repo / "source.txt").write_text("original\n")
    prepare_runtime.run("git", "add", ".", cwd=repo)
    prepare_runtime.run("git", "commit", "-m", "Initial source", cwd=repo)
    return repo, {"repository": str(repo), "revision": prepare_runtime.run("git", "rev-parse", "HEAD", cwd=repo)}


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("patched", [False, True])
def test_reconstruct_sources(tmp_path, source_repo, use_cache, patched):
    repo, source = source_repo
    if patched:
        (repo / "source.txt").write_text("updated\n")
        patch = tmp_path / "runtime/miles/change.patch"
        patch.parent.mkdir(parents=True)
        delta = subprocess.check_output(["git", "diff", "--binary", "--full-index", "HEAD"], cwd=repo)
        patch.write_bytes(delta)
        source.update(patch="change.patch", patch_sha256=hashlib.sha256(delta).hexdigest())
    target = tmp_path / "prepared"
    prepare_runtime.prepare_source("example", source, target, root=tmp_path, cache=repo if use_cache else None)
    shutil.rmtree(repo)
    assert (target / "source.txt").read_text() == ("updated\n" if patched else "original\n")
    assert prepare_runtime.run("git", "rev-parse", "HEAD", cwd=target) == source["revision"]
    assert prepare_runtime.run("git", "show", "HEAD:source.txt", cwd=target) == "original"
    assert not (target / ".git/objects/info/alternates").exists()
    assert bool(prepare_runtime.run("git", "diff", "--cached", cwd=target)) == patched
    with pytest.raises(FileExistsError):
        prepare_runtime.prepare_source("example", source, target, root=tmp_path)


def test_bad_patch_checksum_fails_before_fetch(tmp_path, source_repo):
    _, source = source_repo
    patch = tmp_path / "runtime/miles/change.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("not a valid patch")
    source.update(patch="change.patch", patch_sha256="incorrect")
    target = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Patch checksum mismatch"):
        prepare_runtime.prepare_source("example", source, target, root=tmp_path)
    assert not target.exists()


def test_private_fetch_credentials_are_ephemeral(tmp_path, source_repo, monkeypatch):
    _, source = source_repo
    source["private"] = True
    token_file = tmp_path / "token"
    token_file.write_text("test-token\n")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Existing")
    calls = []
    original_run = prepare_runtime.run

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(prepare_runtime, "run", record)
    target = tmp_path / "prepared"
    prepare_runtime.prepare_source("example", source, target, root=tmp_path, token_file=token_file)
    fetch_args, fetch_kwargs = next(call for call in calls if call[0][:2] == ("git", "fetch"))
    env = fetch_kwargs["env"]
    encoded = base64.b64encode(b"x-access-token:test-token").decode()
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"] == f"AUTHORIZATION: basic {encoded}"
    assert "test-token" not in str(fetch_args)
    assert all(not kwargs.get("env") for args, kwargs in calls if args[:2] != ("git", "fetch"))
    config = (target / ".git/config").read_text()
    assert "test-token" not in config and encoded not in config and "extraheader" not in config


@pytest.mark.parametrize("credential_source", ["GH_TOKEN", "GITHUB_TOKEN", "gh"])
def test_image_build_passes_token_as_secret(monkeypatch, credential_source):
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    if credential_source != "gh":
        monkeypatch.setenv(credential_source, "test-token")
    monkeypatch.setattr(sys, "argv", ["build_image.py", "--base-image", "base", "--tag", "result"])

    def output(args, **kwargs):
        if args[:3] == ["docker", "image", "inspect"]:
            return lock["base_image"]["docker_id"]
        if args[:2] == ["git", "rev-parse"]:
            return "application-revision"
        assert args == ["gh", "auth", "token"] and credential_source == "gh"
        return "test-token\n"

    calls = []
    monkeypatch.setattr(build_image.subprocess, "check_output", output)
    monkeypatch.setattr(build_image.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    build_image.main()
    args, kwargs = calls[0]
    assert args[args.index("--secret") + 1] == "id=github_token,env=MILES_BUILD_GITHUB_TOKEN"
    assert "test-token" not in str(args)
    assert kwargs["env"]["MILES_BUILD_GITHUB_TOKEN"] == "test-token"


@pytest.mark.parametrize("mode_matrix", [False, True])
def test_serving_launcher_uses_merged_tool(mode_matrix):
    spec = launch_hero_serving.specification("image", hf="/model with spaces", mode_matrix=mode_matrix)
    command = spec["tasks"][0]["arguments"][0]
    assert "tools/qualify_serving.py" in command
    assert "--model '/model with spaces'" in command
    assert ("--diagnostic-mode-matrix" in command) == mode_matrix
    assert "--core-reference" not in command
    assert all(dataset["mountPath"] != "/reference" for dataset in spec["tasks"][0]["datasets"])
