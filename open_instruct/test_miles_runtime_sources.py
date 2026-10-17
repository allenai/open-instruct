"""Runtime source preparation fetches immutable commits without modifying them."""

import base64
import json
import shutil
import sys
from pathlib import Path

import pytest
from scripts.miles import build_image, prepare_runtime


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
def test_prepare_sources(tmp_path, source_repo, use_cache):
    repo, source = source_repo
    # A changed branch tip and dirty checkout must not change the selected source.
    (repo / "source.txt").write_text("newer commit\n")
    prepare_runtime.run("git", "commit", "-am", "Advance branch", cwd=repo)
    (repo / "source.txt").write_text("uncommitted\n")
    target = tmp_path / "prepared"
    prepare_runtime.prepare_source("example", source, target, cache=repo if use_cache else None)
    shutil.rmtree(repo)
    assert (target / "source.txt").read_text() == "original\n"
    assert prepare_runtime.run("git", "rev-parse", "HEAD", cwd=target) == source["revision"]
    assert prepare_runtime.run("git", "show", "HEAD:source.txt", cwd=target) == "original"
    assert not (target / ".git/objects/info/alternates").exists()
    assert prepare_runtime.run("git", "status", "--porcelain", cwd=target) == ""
    with pytest.raises(FileExistsError):
        prepare_runtime.prepare_source("example", source, target)


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
    prepare_runtime.prepare_source("example", source, target, token_file=token_file)
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
    monkeypatch.setattr(
        sys, "argv", ["build_image.py", "--base-image", "base", "--tag", "result", "--flash-attention-max-jobs", "1"]
    )

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
    assert args[args.index("--platform") + 1] == "linux/amd64"
    assert args[args.index("--secret") + 1] == "id=github_token,env=MILES_BUILD_GITHUB_TOKEN"
    assert f"RUNTIME_INPUTS={build_image.runtime_inputs_digest(root)}" in args
    assert "FLASH_ATTENTION_MAX_JOBS=1" in args
    assert "test-token" not in str(args)
    assert kwargs["env"]["MILES_BUILD_GITHUB_TOKEN"] == "test-token"


def test_runtime_inputs_digest_ignores_only_the_application_stage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for relative in ["runtime/miles", "scripts/miles/prepare_runtime.py"]:
        source, target = root / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if source.is_dir() else shutil.copy)(source, target)
    digest = build_image.runtime_inputs_digest(tmp_path)
    assert digest == build_image.runtime_inputs_digest(root)
    dockerfile = tmp_path / "runtime/miles/Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + "RUN true\n")
    assert build_image.runtime_inputs_digest(tmp_path) == digest
    dockerfile.write_text(dockerfile.read_text().replace("FROM runtime-base AS application", "RUN true\n&"))
    assert build_image.runtime_inputs_digest(tmp_path) != digest
    for changed in ["runtime/miles/runtime.lock.json", "scripts/miles/prepare_runtime.py"]:
        copy = tmp_path / changed
        original = copy.read_text()
        copy.write_text(original + " ")
        assert build_image.runtime_inputs_digest(tmp_path) != build_image.runtime_inputs_digest(root)
        copy.write_text(original)


def run_runtime_base_build(monkeypatch, labels):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["build_image.py", "--runtime-base", "published-base", "--tag", "result"])

    def output(args, **kwargs):
        if args[:3] == ["docker", "image", "inspect"]:
            assert args[3] == "published-base"
            return json.dumps(labels(build_image.runtime_inputs_digest(root)))
        assert args[:2] == ["git", "rev-parse"], "an application build on a runtime base needs no GitHub token"
        return "application-revision"

    calls = []
    monkeypatch.setattr(build_image.subprocess, "check_output", output)
    monkeypatch.setattr(build_image.subprocess, "run", lambda args, **kwargs: calls.append(args))
    build_image.main()
    return calls[0]


def test_application_build_reuses_a_matching_runtime_base(monkeypatch):
    args = run_runtime_base_build(monkeypatch, lambda digest: {build_image.RUNTIME_INPUTS_LABEL: digest})
    assert args[args.index("--build-context") + 1] == "runtime-base=docker-image://published-base"
    assert args[args.index("--target") + 1] == "application"
    assert "--secret" not in args and not any(arg.startswith("BASE_IMAGE=") for arg in args)


@pytest.mark.parametrize(
    "labels",
    [
        lambda digest: {build_image.RUNTIME_INPUTS_LABEL: "0" * 64},
        lambda digest: None,
        lambda digest: {build_image.RUNTIME_INPUTS_LABEL: digest, build_image.REVISION_LABEL: "abc"},
    ],
    ids=["other-inputs", "unlabelled", "application-image"],
)
def test_application_build_refuses_other_runtime_bases(monkeypatch, labels):
    with pytest.raises(ValueError, match="runtime"):
        run_runtime_base_build(monkeypatch, labels)
