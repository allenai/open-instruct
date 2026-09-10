"""Offline canonical selection, prompt equivalence, and frozen-file checks."""

import copy
import json
from types import SimpleNamespace

import pytest
from scripts.miles import prepare_gsm8k_parity as prep


class Tokenizer:
    chat_template = "fixture template"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = "".join(f"{row['role']}: {row['content']}\n" for row in messages) + "assistant: "
        return list(text.encode()) if tokenize else text


def native_row():
    return {
        "id": "gsm8k-train-000001",
        "messages": [{"role": "user", "content": "How many?"}],
        "ground_truth": "42",
        "metadata": {
            "prepared_sample_id": "gsm8k-train:1:gsm8k-train-000001",
            "verifiers": [{"name": "gsm8k", "target": "42", "weight": 1.0}],
            "source": {"index": 1, "split": "train"},
        },
    }


def test_derive_preserves_canonical_identity_without_mutation():
    row = native_row()
    before = copy.deepcopy(row)
    derived, evidence = prep.derive_rows([row], Tokenizer())
    assert row == before
    assert derived[0]["metadata"]["prepared_sample_id"] == row["metadata"]["prepared_sample_id"]
    assert derived[0]["label"] == "42"
    assert evidence[0]["prompt_tokens"] == len(derived[0]["input"].encode())


@pytest.mark.parametrize("failure", ["reference", "label", "verifier", "overlong", "token_ids", "duplicate"])
def test_derive_rejects_contract_mismatch(failure):
    row = native_row()
    tokenizer = Tokenizer()
    maximum = 2048
    if failure == "reference":
        row["messages"].insert(0, {"role": "assistant", "content": "REFERENCE"})
    elif failure == "label":
        row["ground_truth"] = ["42"]
    elif failure == "verifier":
        row["metadata"]["verifiers"][0]["target"] = "43"
    elif failure == "overlong":
        maximum = 2
    elif failure == "token_ids":
        tokenizer.encode = lambda *args, **kwargs: [999]
    with pytest.raises(ValueError):
        prep.derive_rows([row, row] if failure == "duplicate" else [row], tokenizer, maximum)


def test_canonical_preparer_selects_once_and_derives_both_arms(tmp_path, monkeypatch):
    baseline = pytest.importorskip("olmo_miles.rl.rl_prepare")
    calls = []

    def load_source(spec, **kwargs):
        calls.append(spec.name)
        return [
            {
                "messages": [
                    {"role": "user", "content": f"{spec.split} question {index}?"},
                    {"role": "assistant", "content": "SECRET REFERENCE"},
                ],
                "ground_truth": ["1,234"],
            }
            for index in range(500 if spec.split == "train" else 200)
        ], dict(spec.source)

    def descriptor(root):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            prep.write_immutable(root / "hf" / name, b"{}")
        prep.write_immutable(root / "hf/chat_template.jinja", Tokenizer.chat_template.encode())
        return {"source": "offline fixture", "template_sha256": prep.TEMPLATE_SHA256}

    real_import = prep.importlib.import_module
    monkeypatch.setattr(baseline, "_load_source", load_source)
    monkeypatch.setattr(prep, "prepare_descriptor", descriptor)
    monkeypatch.setattr(prep, "TEMPLATE_SHA256", prep.digest(Tokenizer.chat_template.encode()))
    monkeypatch.setattr(
        prep.importlib,
        "import_module",
        lambda name: SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
        if name == "transformers"
        else real_import(name),
    )
    report = prep.prepare(tmp_path)
    assert calls == ["gsm8k-train", "gsm8k-eval"]
    assert report["partitions"]["train"]["records"] == 400
    assert report["partitions"]["eval"]["records"] == 128
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert all("SECRET REFERENCE" not in row["input"] and row["label"] == "1234" for row in train)
    selected = [int(row["metadata"]["prepared_sample_id"].split(":")[1]) for row in train]
    assert selected != list(range(400)) and len(set(selected)) == 400
    assert prep.prepare(tmp_path) == report
    assert calls == ["gsm8k-train", "gsm8k-eval"]  # Cache verification must not download/reselect.
    (tmp_path / "eval.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="SHA256 differs"):
        prep.verify_preparation(tmp_path)


def test_immutable_artifacts_reject_replacement(tmp_path):
    path = tmp_path / "train.jsonl"
    prep.write_immutable(path, b"first\n")
    prep.write_immutable(path, b"first\n")
    with pytest.raises(ValueError, match="Refusing to replace"):
        prep.write_immutable(path, b"second\n")
