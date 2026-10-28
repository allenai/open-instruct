"""The dependency base must never build against a silently substituted toolchain."""

import json
import sys

import pytest
from scripts.miles import build_base_image


@pytest.mark.parametrize("mismatch", [None, "compiled_parent", "flash_attention_builder"])
def test_base_builder_validates_both_parents_before_build(tmp_path, monkeypatch, mismatch):
    lock = {
        "compiled_parent": {"docker_id": "compiled-digest"},
        "flash_attention_builder": {"docker_id": "builder-digest"},
    }
    path = tmp_path / "runtime/miles/base-build.lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(build_base_image, "__file__", str(tmp_path / "scripts/miles/build_base_image.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_base_image.py",
            "--compiled-parent",
            "compiled_parent",
            "--flash-attention-builder",
            "flash_attention_builder",
            "--tag",
            "result",
        ],
    )
    builds = []

    def inspect(args, **kwargs):
        name = args[3]
        return "unexpected-digest" if name == mismatch else lock[name]["docker_id"]

    monkeypatch.setattr(build_base_image.subprocess, "check_output", inspect)
    monkeypatch.setattr(build_base_image.subprocess, "run", lambda args, **kwargs: builds.append(args))
    if mismatch:
        with pytest.raises(ValueError, match=mismatch):
            build_base_image.main()
        assert not builds
    else:
        build_base_image.main()
        assert len(builds) == 1
        assert "runtime/miles/Dockerfile.base" in builds[0]
        assert "COMPILED_PARENT=compiled_parent" in builds[0]
        assert "COMPILED_PARENT_DIGEST=compiled-digest" in builds[0]
        assert "FLASH_ATTENTION_BUILDER=flash_attention_builder" in builds[0]
