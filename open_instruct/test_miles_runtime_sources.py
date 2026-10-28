"""Runtime source preparation fetches immutable commits without modifying them."""

import json
import shutil
import sys

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


@pytest.mark.parametrize("target", ["runtime-base", "application"])
@pytest.mark.parametrize("base_matches", [True, False])
def test_image_build_validates_base_and_records_revision(tmp_path, monkeypatch, target, base_matches):
    lock_path = tmp_path / "runtime/miles/runtime.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"base_image": {"docker_id": "base-digest"}}))
    monkeypatch.setattr(build_image, "__file__", str(tmp_path / "scripts/miles/build_image.py"))
    monkeypatch.setattr(sys, "argv", ["build_image.py", "--base-image", "base", "--tag", "result", "--target", target])

    def output(args, **kwargs):
        if args[:3] == ["docker", "image", "inspect"]:
            return "base-digest" if base_matches else "other-digest"
        assert args == ["git", "rev-parse", "HEAD"]
        return "application-revision"

    calls = []
    monkeypatch.setattr(build_image.subprocess, "check_output", output)
    monkeypatch.setattr(build_image.subprocess, "run", lambda args, **kwargs: calls.append(args))
    if not base_matches:
        with pytest.raises(ValueError, match="Base image differs"):
            build_image.main()
        assert not calls
        return
    build_image.main()
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("--platform") + 1] == "linux/amd64"
    assert args[args.index("--target") + 1] == target
    assert "BASE_IMAGE=base" in args
    assert "SOURCE_REVISION=application-revision" in args
    assert "--secret" not in args
