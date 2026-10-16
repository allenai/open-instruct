import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from open_instruct import dataset_transformation, resolved_sft_dataset


@dataclasses.dataclass
class Source:
    dataset_name: str = "example"
    dataset_commit_hash: str = "pinned-commit"
    dataset: object = None


@dataclasses.dataclass
class Tokenizer:
    tokenizer_files_hash: str = "tokenizer-hash"

    @property
    def tokenizer(self):
        return SimpleNamespace(chat_template="chat-template")


def test_historical_hash_formula_is_unchanged():
    dcs, tc = [Source()], Tokenizer()
    record = {"dataset_name": "example", "dataset_commit_hash": "pinned-commit"}
    payload = {
        "cache_version": dataset_transformation.DATASET_CACHE_VERSION,
        "dataset_configs": [record],
        "tokenizer_config": dataclasses.asdict(tc),
        "chat_template_hash": hashlib.sha256(json.dumps("chat-template", sort_keys=True).encode()).hexdigest(),
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    assert dataset_transformation.compute_config_hash(dcs, tc) == expected
    assert dataset_transformation.compute_config_hash_from_dicts([record], tc) == expected


def test_manifest_pins_request_tokenizer_and_resolved_revision(tmp_path):
    path = tmp_path / "dataset.json"
    request = {"dataset_mixer_list": ["example", "1.0"]}
    resolved_sft_dataset.write(path, request, [Source()], Tokenizer())
    expected = dataset_transformation.compute_config_hash([Source()], Tokenizer())
    actual, receipt = resolved_sft_dataset.read(path, request, Tokenizer())
    assert actual == expected and len(receipt) == 64
    with pytest.raises(ValueError, match="request differs"):
        resolved_sft_dataset.read(path, {"dataset_mixer_list": ["example", "0.5"]}, Tokenizer())
    with pytest.raises(ValueError, match="cache hash"):
        resolved_sft_dataset.read(path, request, Tokenizer("different-tokenizer"))
    manifest = json.loads(path.read_text())
    manifest["dataset_configs"][0]["dataset_commit_hash"] = "different-revision"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cache hash"):
        resolved_sft_dataset.read(path, request, Tokenizer())


def test_different_manifest_cannot_overwrite_verified_metadata(tmp_path):
    path = tmp_path / "dataset.json"
    resolved_sft_dataset.write(path, {}, [Source()], Tokenizer())
    resolved_sft_dataset.write(path, {}, [Source()], Tokenizer())
    with pytest.raises(FileExistsError, match="Refusing"):
        resolved_sft_dataset.write(path, {}, [Source(dataset_name="other")], Tokenizer())
