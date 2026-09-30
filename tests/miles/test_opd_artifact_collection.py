import tarfile

from scripts.miles import collect_opd_selection_artifacts


def test_collector_keeps_large_eval_but_not_checkpoint_weights(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "output"
    evaluation = root / "drop-8u/eval-capture/eval-7.jsonl"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_bytes(b" " * (33 * 1024 * 1024))
    checkpoint = root / "drop-8u/hf-7"
    checkpoint.mkdir()
    (checkpoint / ".complete").touch()
    (checkpoint / "model.safetensors").write_bytes(b"excluded")
    manifest = collect_opd_selection_artifacts.collect(root, output)
    assert manifest["arms"]["drop-8u"]["final_eval_exists"]
    assert not manifest["arms"]["retry-8u"]["final_eval_exists"]
    with tarfile.open(output / "opd-selection-artifacts.tar.gz") as archive:
        assert archive.getnames() == ["drop-8u/eval-capture/eval-7.jsonl", "drop-8u/hf-7/.complete"]
    assert manifest["files"][0]["bytes"] == 33 * 1024 * 1024
