import pytest
from scripts.train.atomic_think import run


def test_diagnostic_reuses_packing_with_independent_loader_directory(tmp_path):
    original = tmp_path / "gate-reference" / "dataset-common"
    original.mkdir(parents=True)
    (original / "indices.npy").write_bytes(b"verified packing")
    run.link_reference_packing(tmp_path, "gate-reference-repeat")
    run.link_reference_packing(tmp_path, "gate-reference-repeat")
    repeat = tmp_path / "gate-reference-repeat"
    assert (repeat / "dataset-common" / "indices.npy").read_bytes() == b"verified packing"
    (repeat / "loader-state").write_bytes(b"independent")
    assert not (original.parent / "loader-state").exists()


def test_diagnostic_rejects_missing_or_different_packing(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.link_reference_packing(tmp_path, "gate-reference-repeat")
    (tmp_path / "gate-reference" / "dataset-common").mkdir(parents=True)
    target = tmp_path / "gate-reference-repeat" / "dataset-common"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="differs from reference"):
        run.link_reference_packing(tmp_path, "gate-reference-repeat")
