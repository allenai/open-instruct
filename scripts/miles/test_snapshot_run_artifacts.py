import hashlib
import tarfile

import pytest
from scripts.miles import snapshot_run_artifacts


def test_preserves_large_logs_binary_payloads_and_records_external_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rollout.pt").write_bytes(b"\x00\xff\x01" * 100)
    (source / "large.log").write_bytes(b"x" * (33 * 1024 * 1024))
    external = tmp_path / "external-model"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"must not be copied")
    (source / "prepared-hf").symlink_to(external, target_is_directory=True)
    output = tmp_path / "snapshot"
    manifest = snapshot_run_artifacts.snapshot(source, output)
    assert manifest["file_count"] == 2
    assert manifest["symlinks"][0]["target"] == str(external)
    with tarfile.open(output / manifest["archive"]["path"]) as archive:
        assert set(archive.getnames()) == {"run/rollout.pt", "run/large.log"}
        for row in manifest["files"]:
            with archive.extractfile("run/" + row["path"]) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]


def test_source_mutation_rejects_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "run.log"
    path.write_text("before")
    original = tarfile.TarFile.addfile

    def mutate_after_copy(archive, *args, **kwargs):
        result = original(archive, *args, **kwargs)
        path.write_text("changed during copy")
        return result

    monkeypatch.setattr(tarfile.TarFile, "addfile", mutate_after_copy)
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="changed during snapshot"):
        snapshot_run_artifacts.snapshot(source, output)
    assert not (output / "snapshot-manifest.json").exists()
    assert (output / "snapshot-failure.json").is_file()


def test_rejects_recursive_output_and_existing_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        snapshot_run_artifacts.snapshot(source, source / "snapshot")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        snapshot_run_artifacts.snapshot(source, output)
    assert (output / "keep").read_text() == "keep"
