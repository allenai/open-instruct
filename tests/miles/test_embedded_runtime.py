"""Private source overlays must match the base marker and full-index patch."""

import subprocess

import pytest
from scripts.miles.prepare_runtime import copy_embedded_source


def test_embedded_source_rejects_wrong_revision_before_copy(tmp_path):
    original, target = tmp_path / "original", tmp_path / "target"
    original.mkdir()
    (original / ".source-revision").write_text("original\n")
    with pytest.raises(ValueError, match="revision mismatch"):
        copy_embedded_source({"embedded_path": str(original), "revision": "wrong"}, target)
    assert not target.exists()


def test_embedded_source_preserves_contents_for_patch_verification(tmp_path):
    original, target = tmp_path / "original", tmp_path / "target"
    original.mkdir()
    (original / ".source-revision").write_text("original\n")
    (original / "model.py").write_text("value = 1\n")
    copy_embedded_source({"embedded_path": str(original), "revision": "original"}, target)
    assert (target / "model.py").read_text() == "value = 1\n"
    (target / "model.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "model.py"], cwd=target, check=True)
    patch = subprocess.check_output(["git", "diff", "--cached", "--full-index", "--binary", "HEAD"], cwd=target)
    assert b"-value = 1" in patch and b"+value = 2" in patch
    assert (original / "model.py").read_text() == "value = 1\n"
