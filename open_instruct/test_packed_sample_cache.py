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
