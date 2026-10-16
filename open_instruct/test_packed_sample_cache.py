import json

import numpy as np
import pytest
import torch
from olmo_core import data

from open_instruct import packed_sample_cache


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMO_DATA_PREP_WORKERS", "1")
    tokens = tmp_path / "tokens.npy"
    mask = tmp_path / "mask.npy"
    np.array([3, 4, 2, 5, 6, 7, 2, 8, 2, 9, 10, 2], dtype=np.uint8).tofile(tokens)
    np.array([False, True, True, False, True, True, True, False, True, True, False, True]).tofile(mask)
    ds = data.NumpyPackedFSLDatasetConfig(
        tokenizer=data.TokenizerConfig(vocab_size=32, eos_token_id=2, pad_token_id=0),
        paths=[str(tokens)],
        label_mask_paths=[str(mask)],
        work_dir=str(tmp_path / "work"),
        sequence_length=8,
        generate_doc_lengths=True,
        long_doc_strategy=data.LongDocStrategy.truncate,
    ).build()
    ds.prepare()
    return ds


def test_exact_samples_preserve_fingerprint_collation_and_fallback(dataset, tmp_path):
    original = dataset[0]
    directory = tmp_path / "cache"
    packed_sample_cache.build_cache(dataset, [0], directory, provenance={"test": True}, workers=1)
    cached = packed_sample_cache.CachedPackedFSLDataset(dataset, directory)
    assert cached.fingerprint == dataset.fingerprint and len(cached) == len(dataset)
    collator = data.DataCollator(pad_token_id=0, vocab_size=32)
    for index in range(len(dataset)):
        expected, actual = collator([dataset[index]]), collator([cached[index]])
        assert expected.keys() == actual.keys()
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                assert expected[key].dtype == actual[key].dtype
                assert torch.equal(expected[key], actual[key])
            else:
                assert expected[key] == actual[key]
    assert torch.equal(cached[0]["input_ids"], original["input_ids"])


def test_corrupt_payload_and_wrong_dataset_rejected(dataset, tmp_path):
    directory = tmp_path / "cache"
    packed_sample_cache.build_cache(dataset, [0], directory, provenance={}, workers=1)
    tokens = np.load(directory / "input_ids.npy", mmap_mode="r+")
    tokens[0, 0] += 1
    tokens.flush()
    cached = packed_sample_cache.CachedPackedFSLDataset(dataset, directory)
    with pytest.raises(ValueError, match="integrity failure"):
        cached[0]
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_fingerprint"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another dataset"):
        packed_sample_cache.CachedPackedFSLDataset(dataset, directory)


def test_stage_cache_publishes_only_verified_local_payload(dataset, tmp_path):
    source, destination = tmp_path / "source", tmp_path / "local"
    packed_sample_cache.build_cache(dataset, [0], source, provenance={}, workers=1)
    receipt = packed_sample_cache.stage_cache(source, destination)
    assert receipt["files"] == json.loads((source / "manifest.json").read_text())["files"]
    assert torch.equal(
        packed_sample_cache.CachedPackedFSLDataset(dataset, destination)[0]["input_ids"], dataset[0]["input_ids"]
    )
    with (source / "input_ids.npy").open("ab") as stream:
        stream.write(b"corrupt")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="staging hash mismatch"):
        packed_sample_cache.stage_cache(source, failed)
    assert not (failed / "manifest.json").exists()


def test_interrupted_cache_resumes_verified_rows_without_rereading(dataset, tmp_path, monkeypatch):
    directory = tmp_path / "resumable"
    original = packed_sample_cache.bounded_samples

    def interrupted(dataset, indices, workers):
        yield 0, dataset[indices[0]]
        raise RuntimeError("interrupted")

    monkeypatch.setattr(packed_sample_cache, "bounded_samples", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        packed_sample_cache.build_cache(dataset, [0, 1], directory, provenance={}, checkpoint_interval=1)
    assert not (directory / "manifest.json").exists()
    assert json.loads((directory / "progress.json").read_text())["rows"][0] is not None
    with pytest.raises(ValueError, match="sample selection"):
        packed_sample_cache.build_cache(dataset, [1, 0], directory, provenance={}, resume=True)
    tokens = np.load(directory / "input_ids.npy", mmap_mode="r+")
    tokens[0, 0] += 1
    tokens.flush()
    with pytest.raises(ValueError, match="integrity verification"):
        packed_sample_cache.build_cache(dataset, [0, 1], directory, provenance={}, resume=True)
    tokens[0, 0] -= 1
    tokens.flush()
    observed = []

    def remaining(dataset, indices, workers):
        observed.extend(indices)
        yield from original(dataset, indices, workers)

    monkeypatch.setattr(packed_sample_cache, "bounded_samples", remaining)
    packed_sample_cache.build_cache(dataset, [0, 1], directory, provenance={}, resume=True)
    assert observed == [1]
    cached = packed_sample_cache.CachedPackedFSLDataset(dataset, directory)
    for index in (0, 1):
        assert torch.equal(cached[index]["input_ids"], dataset[index]["input_ids"])
