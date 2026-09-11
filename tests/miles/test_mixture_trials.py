"""Exercise deterministic selection, real MILES grouping, and all three verifiers."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from miles.rollout.data_source import RolloutDataSource
from miles.utils.data import Dataset
from miles.utils.types import Sample
from scripts.miles import mixture_trials as trial

from open_instruct.miles import mixture


def tokenizer():
    return SimpleNamespace(chat_template="fixture chat", encode=lambda text, **kwargs: [ord(c) % 50 + 3 for c in text])


def source_snapshots(root, shared_prompts=False):
    roots = {}
    for source, verifier in mixture.SOURCES.items():
        folder = root / source
        folder.mkdir(parents=True)
        roots[source] = folder
        hashes, raw_source = {}, b""
        for split, count in (("train", 8), ("eval", 4)):
            target = (
                "-2"
                if source == "gsm8k"
                else "42"
                if source == "math"
                else json.dumps({"func_name": "validate_lowercase"})
            )
            rows = [
                {
                    "input": "common training prompt"
                    if shared_prompts and split == "train"
                    else f"{source} {split} question {i}",
                    "label": target,
                    "metadata": {
                        "prepared_sample_id": f"{split}-{i}",
                        "source_row": i,
                        "source_revision": "a" * 40,
                        "query": f"Question {i}",
                        "verifiers": [{"name": verifier, "target": target}],
                    },
                }
                for i in range(count)
            ]
            raw = b"".join(mixture.encoded(row) for row in rows)
            raw_source += raw
            (folder / f"{split}.jsonl").write_bytes(raw)
            hashes[split] = mixture.digest(raw)
        (folder / "source.jsonl").write_bytes(raw_source)
        (folder / "preparation.json").write_bytes(
            mixture.encoded(
                {
                    "template_sha256": mixture.digest(tokenizer().chat_template.encode()),
                    "prepared_sha256": hashes,
                    "source": {"kind": "local_jsonl", "sha256": mixture.digest(raw_source)},
                }
            )
        )
    return roots


def prepare_fixture(root, monkeypatch, shared_prompts=False):
    sources = source_snapshots(root / "sources", shared_prompts=shared_prompts)
    monkeypatch.setattr(trial.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer())
    hf = root / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
            }
        )
    )
    output = root / "mixture"
    trial.prepare(output, sources, root / "hf", local=True)
    return output, sources


def real_groups(root):
    source = RolloutDataSource(
        SimpleNamespace(rollout_global_dataset=False, rollout_shuffle=False, n_samples_per_prompt=4)
    )
    source.dataset = Dataset(str(root / "train.jsonl"), tokenizer(), None, None, prompt_key="input", label_key="label")
    return [source.get_samples(trial.BATCH_PROMPTS) for _ in range(mixture.UPDATES)]


def capture(sample, *, correct, version):
    sample = copy.deepcopy(sample)
    source = sample.metadata["mixture"]["source"]
    response = {
        "gsm8k": "The final answer is -2" if correct else "The final answer is 3",
        "math": r"\boxed{42}" if correct else r"\boxed{43}",
        "ifeval": "all lowercase" if correct else "Uppercase",
    }[source]
    sample.response = response
    sample.response_length = 1
    sample.tokens = tokenizer().encode(sample.prompt) + [1]
    sample.reward = float(correct)
    sample.status = Sample.Status.COMPLETED
    sample.rollout_log_probs = [-0.5]
    sample.weight_versions = [str(version)]
    return sample.to_dict()


def captured_run(root):
    _, rows = mixture.verify(root)
    (root / "rollouts").mkdir()
    (root / "metrics").mkdir()
    for update, groups in enumerate(real_groups(root)):
        samples = [
            capture(sample, correct=i % 2 == 0, version=update) for group in groups for i, sample in enumerate(group)
        ]
        torch.save({"rollout_id": update, "samples": samples}, root / f"rollouts/{update}.pt")
    for name, version in (("eval_0", 0), ("eval_1", 2)):
        samples = [
            capture(
                Sample(prompt=row["input"], label=row["label"], metadata=row["metadata"]),
                correct=True,
                version=version,
            )
            for row in rows["eval"]
        ]
        torch.save({"rollout_id": int(name[-1]), "samples": samples}, root / f"rollouts/{name}.pt")
    (root / "arguments.json").write_text(json.dumps(trial.configuration(root).arguments()))
    (root / "metrics/training_contract_rank0.jsonl").write_text(
        ' {"event":"optimizer","step":1}\n{"event":"optimizer","step":2}\n'
    )
    (root / "metrics/publication.jsonl").write_text(
        "".join(
            json.dumps({"version": version, "repeated_version": repeated}) + "\n"
            for version, repeated in ((0, False), (1, False), (1, True), (2, False), (2, True))
        )
    )


def test_deterministic_balanced_selection_preserves_original_metadata(tmp_path, monkeypatch):
    root, sources = prepare_fixture(tmp_path, monkeypatch)
    before = {name: (path / "train.jsonl").read_bytes() for name, path in sources.items()}
    second = tmp_path / "same"
    trial.prepare(second, dict(reversed(list(sources.items()))), tmp_path / "hf", local=True)
    assert (root / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    report, rows = trial.verify_preparation(root)
    assert len(rows["train"]) == 12 and len(rows["eval"]) == 6
    for batch in real_groups(root):
        assert len(batch) == 6 and all(len(group) == 4 for group in batch)
        assert Counter(group[0].metadata["mixture"]["source"] for group in batch) == {
            name: 2 for name in mixture.SOURCES
        }
        assert len({group[0].group_index for group in batch}) == 6
        for group in batch:
            assert len({sample.group_index for sample in group}) == 1
            assert len({sample.metadata["mixture"]["id"] for sample in group}) == 1
    for row in rows["train"]:
        marker = row["metadata"]["mixture"]
        original = json.loads(
            sources[marker["source"]].joinpath("train.jsonl").read_text().splitlines()[marker["source_position"]]
        )
        assert {k: v for k, v in row["metadata"].items() if k != "mixture"} == original["metadata"]
        assert marker["source_row_sha256"] == mixture.digest(mixture.encoded(original))
    assert before == {name: (path / "train.jsonl").read_bytes() for name, path in sources.items()}
    assert len(set(report["schedule"][0]) & set(report["schedule"][1])) == 0


def test_real_three_source_reward_audit_and_local_parser(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path, monkeypatch)
    captured_run(root)
    result = trial.audit(root)
    assert result["passed"]
    for source in mixture.SOURCES:
        assert result["results"]["0"]["sources"][source] == {
            "samples": 8,
            "mean_reward": 0.5,
            "prompt_groups": 2,
            "mixed_reward_groups": 2,
            "responses_at_token_cap": 0,
        }
        assert result["results"]["eval_1"]["sources"][source]["samples"] == 2
    validated = trial.run(root, validate_only=True)
    assert validated == {"validated": True, "prompts_per_update": 6, "responses_per_update": 24}


@pytest.mark.parametrize("fault", ["target", "source", "group", "membership", "tokens", "version", "reward"])
def test_mixture_rejects_cross_source_or_stale_evidence(tmp_path, monkeypatch, fault):
    root, _ = prepare_fixture(tmp_path, monkeypatch)
    captured_run(root)
    path = root / "rollouts/0.pt"
    payload = torch.load(path, weights_only=False)
    sample = payload["samples"][0]
    if fault == "target":
        sample["metadata"]["verifiers"][0]["target"] = "999"
    elif fault == "source":
        sample["metadata"]["mixture"]["source"] = "math"
    elif fault == "group":
        sample["group_index"] = 999
    elif fault == "membership":
        payload["samples"].pop()
    elif fault == "tokens":
        sample["tokens"][0] += 1
    elif fault == "version":
        sample["weight_versions"] = ["1"]
    elif fault == "reward":
        sample["reward"] = 0.0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        trial.audit(root)


def test_rejects_changed_parent_snapshot_and_mixture_registry(tmp_path, monkeypatch):
    root, sources = prepare_fixture(tmp_path, monkeypatch)
    (sources["math"] / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed prepared"):
        trial.prepare(tmp_path / "bad", sources, tmp_path / "hf", local=True)
    (root / "verifiers.json").write_text("{}")
    with pytest.raises(ValueError, match="registry changed"):
        trial.verify_preparation(root)


def test_identical_prompt_text_never_merges_source_groups(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path, monkeypatch, shared_prompts=True)
    groups = [group for batch in real_groups(root) for group in batch]
    assert len({sample.prompt for group in groups for sample in group}) == 1
    assert len({group[0].group_index for group in groups}) == 12
    assert len({group[0].metadata["mixture"]["id"] for group in groups}) == 12
    assert all(len(group) == 4 for group in groups)


@pytest.mark.parametrize("fault", ["template", "provenance", "duplicate_identity"])
def test_source_contract_rejects_incompatible_preparation(tmp_path, monkeypatch, fault):
    _, sources = prepare_fixture(tmp_path, monkeypatch)
    path = sources["math"] / "preparation.json"
    manifest = json.loads(path.read_text())
    if fault == "template":
        manifest["template_sha256"] = "different"
    elif fault == "provenance":
        manifest["source"] = {"kind": "huggingface", "dataset": "unknown", "revision": "main"}
    else:
        data = sources["math"] / "train.jsonl"
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        rows[-1]["metadata"]["prepared_sample_id"] = rows[0]["metadata"]["prepared_sample_id"]
        raw = b"".join(mixture.encoded(row) for row in rows)
        data.write_bytes(raw)
        manifest["prepared_sha256"]["train"] = mixture.digest(raw)
    path.write_bytes(mixture.encoded(manifest))
    with pytest.raises(ValueError):
        trial.prepare(tmp_path / "rejected", sources, tmp_path / "hf", local=True)


def test_longer_response_budget_updates_training_serving_and_admission(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path, monkeypatch)
    report = json.loads((root / "preparation.json").read_text())
    report.update(profile="sft", response_cap=8192)
    (root / "preparation.json").write_bytes(mixture.encoded(report))
    config = trial.configuration(root)
    assert config.core.max_sequence_length == 10240
    assert config.miles["rollout_max_context_len"] == config.miles["sglang_context_length"] == 10240
    assert config.miles["rollout_max_response_len"] == config.miles["eval_max_response_len"] == 8192
    assert config.miles["sglang_max_total_tokens"] >= 4 * 10240
    assert config.miles["global_batch_size"] == 24


def test_bad_prepared_response_budget_rejected(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path, monkeypatch)
    report = json.loads((root / "preparation.json").read_text())
    report["response_cap"] = -1
    (root / "preparation.json").write_bytes(mixture.encoded(report))
    with pytest.raises(ValueError, match="response cap"):
        trial.configuration(root)
