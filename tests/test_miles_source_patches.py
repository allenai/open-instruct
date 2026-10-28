"""Explicit runtime patches preserve the old image without changing source caches."""

import pytest
from scripts.miles import prepare_runtime


def test_ordered_runtime_patches_preserve_cache_and_exclude_tests(tmp_path):
    upstream = tmp_path / "upstream"
    prepare_runtime.run("git", "init", str(upstream))
    prepare_runtime.run("git", "config", "user.email", "test@example.com", cwd=upstream)
    prepare_runtime.run("git", "config", "user.name", "Test", cwd=upstream)
    (upstream / "value.txt").write_text("original\n")
    (upstream / "tests").mkdir()
    (upstream / "tests/value.txt").write_text("original test\n")
    prepare_runtime.run("git", "add", ".", cwd=upstream)
    prepare_runtime.run("git", "commit", "-m", "Original", cwd=upstream)
    revision = prepare_runtime.run("git", "rev-parse", "HEAD", cwd=upstream)
    destination = tmp_path / "runtime"
    patch_source = destination / "patches"
    patch_source.mkdir(parents=True)
    for index, value in enumerate(("first", "second")):
        (upstream / "value.txt").write_text(value + "\n")
        (upstream / "tests/value.txt").write_text(value + " test\n")
        (patch_source / f"{index}.patch").write_text(prepare_runtime.run("git", "diff", cwd=upstream) + "\n")
        prepare_runtime.run("git", "commit", "-am", value, cwd=upstream)
    target = destination / "target"
    prepare_runtime.prepare_source("target", {"revision": revision}, target, cache=upstream)
    groups = [{"target": "target", "source": "patches", "files": ["0.patch", "1.patch"], "exclude": ["tests/*"]}]
    prepare_runtime.apply_source_patches(destination, groups)
    assert (target / "value.txt").read_text() == "second\n"
    assert (target / "tests/value.txt").read_text() == "original test\n"
    assert prepare_runtime.run("git", "rev-parse", "HEAD", cwd=target) == revision
    assert prepare_runtime.run("git", "status", "--porcelain", cwd=upstream) == ""
    with pytest.raises(RuntimeError, match="patch"):
        prepare_runtime.apply_source_patches(destination, groups)
    assert (target / "value.txt").read_text() == "second\n"
