"""Cache reuse must preserve prompts, eligibility, pool selection and provenance."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_run_data, test_miles_run_spec
from open_instruct.miles.datasets import preparation_cache, run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rollout import indexed_samples

environment = test_miles_run_data.environment


def config(tmp_path):
    registry = tmp_path / "rewards.json"
    registry.write_text(json.dumps({"math": {"factory": "open_instruct.miles.rewards.math_verifier.MathVerifier"}}))
    sources = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "input": f"{name} question {i}",
                        "label": str(i),
                        "pool_id": "original",
                        "metadata": {
                            "prepared_sample_id": f"{name}-{i}",
                            "verifiers": [{"name": "math", "target": str(i)}],
                        },
                    }
                )
                + "\n"
                for i in range(3)
            )
        )
        sources.append({"name": name, "pool": name, "prompt_data": str(path), "reward_config": str(registry)})
    return {
        "seed": 17,
        "shuffle": True,
        "sources": sources,
        "sampling": {"policy": "uniform"},
        "preparation": {"cache_dir": str(tmp_path / "cache")},
    }


def prepare(env, data, output=None, limit=2000, overrides=None):
    return run_data.prepare_data(
        data, env[0], output or env[1], max_prompt_length=limit, seed=data["seed"], registry_overrides=overrides
    )


def forbid_tokenization(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Reused sources must not instantiate a tokenizer")

    monkeypatch.setattr(run_data, "_tokenizer", fail)


def rows(result):
    return [[json.loads(s) for s in Path(p).read_text().splitlines()] for p in result["train_sources"]]


def test_reuse_across_roots_weights_seed_and_identical_tokenizer_paths(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    original = prepare(environment, data)
    forbid_tokenization(monkeypatch)
    hf = tmp_path / "other-hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_bytes((environment[0] / "tokenizer_config.json").read_bytes())
    data["seed"] = 99
    data["sampling"] = {"policy": "weighted", "weights": {"a": 0.8, "b": 0.2}}
    changed = prepare((hf, tmp_path / "second", None), data)
    assert rows(original) == rows(changed)
    assert all(
        Path(a).resolve() == Path(b).resolve()
        for a, b in zip(original["train_sources"], changed["train_sources"], strict=True)
    )
    assert prepare((hf, tmp_path / "second", None), data) == changed
    manifest = json.loads(Path(changed["manifest"]).read_text())
    assert {r["status"] for r in manifest["provenance"]["cache"]} == {"hit"}


def test_pool_assignment_is_run_metadata_and_does_not_rewrite_source(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    first = prepare(environment, data)
    forbid_tokenization(monkeypatch)
    for s in data["sources"]:
        s["pool"] = "together"
    second = prepare(environment, data, tmp_path / "second")
    recipe = json.loads(Path(second["prompt_mixture"]).read_text())
    catalog = indexed_samples.Catalog(
        Path(second["manifest"]).parent, recipe["sources"], recipe["pools"], sample_type=SimpleNamespace
    )
    try:
        assert set(catalog.pool_codes) == {0}
        assert catalog.sample(0).metadata["sampling_pool"] == "together"
        assert catalog.sample(0).metadata["pool_id"] == "together"
        assert catalog.sample(0).prompt == rows(first)[0][0]["input"]
    finally:
        catalog.close()


def test_verifier_change_reuses_tokens_but_unknown_reference_fails(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    prepare(environment, data)
    forbid_tokenization(monkeypatch)
    registry = Path(data["sources"][0]["reward_config"])
    registry.write_text(json.dumps({"math": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}))
    second = prepare(environment, data, tmp_path / "second")
    assert "GSM8K" in Path(second["reward_config"]).read_text()
    registry.write_text(json.dumps({"wrong": {"factory": "x.Verifier"}}))
    with pytest.raises(InputError, match="unknown/malformed"):
        prepare(environment, data, tmp_path / "third")


def test_length_is_admission_not_cache_identity(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    prepare(environment, data)
    forbid_tokenization(monkeypatch)
    prepare(environment, data, tmp_path / "larger", limit=4000)
    with pytest.raises(InputError, match="exceeds prompt length"):
        prepare(environment, data, tmp_path / "too-short", limit=2)


def test_import_legacy_bundle_preserves_exact_bytes_and_skips_tokenization(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    legacy = copy.deepcopy(data)
    legacy.pop("preparation")
    original = prepare(environment, legacy)
    data["preparation"]["import_manifests"] = [original["manifest"]]
    forbid_tokenization(monkeypatch)
    new = prepare(environment, data, tmp_path / "new")
    assert rows(new) == rows(original)
    assert all(
        Path(a).resolve() == Path(b).resolve()
        for a, b in zip(new["train_sources"], original["train_sources"], strict=True)
    )
    assert {r["status"] for r in json.loads(Path(new["manifest"]).read_text())["provenance"]["cache"]} == {"imported"}


def test_import_fraction_bundle_with_row_labels_into_pooled_run(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    legacy = copy.deepcopy(data)
    legacy.pop("preparation")
    legacy.pop("sampling")
    for source in legacy["sources"]:
        source.pop("pool")
        source["fraction"] = 0.5
    original = prepare(environment, legacy)
    data["preparation"]["import_manifests"] = [original["manifest"]]
    forbid_tokenization(monkeypatch)
    new = prepare(environment, data, tmp_path / "new")
    recipe = json.loads(Path(new["prompt_mixture"]).read_text())
    assert [s["pool_mapping"] for s in recipe["sources"]] == [{"original": "a"}, {"original": "b"}]
    catalog = indexed_samples.Catalog(
        Path(new["manifest"]).parent, recipe["sources"], recipe["pools"], sample_type=SimpleNamespace
    )
    try:
        assert [catalog.sample(row).metadata["sampling_pool"] for row in (0, 3)] == ["a", "b"]
    finally:
        catalog.close()


def test_unused_judge_binding_removal_does_not_invalidate_import(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    legacy = copy.deepcopy(data)
    legacy.pop("preparation")
    binding = {
        "general-quality": {
            "factory": "open_instruct.miles.rewards.judge_registry.NamedJudgeVerifier",
            "config": {"name": "general-quality"},
        }
    }
    original = prepare(environment, legacy, overrides=binding)
    data["preparation"]["import_manifests"] = [original["manifest"]]
    forbid_tokenization(monkeypatch)
    new = prepare(environment, data, tmp_path / "new")
    assert "general-quality" not in json.loads(Path(new["reward_config"]).read_text())


def test_changed_source_only_rebuilds_that_source(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    prepare(environment, data)
    original = run_data._tokenizer
    calls = []
    monkeypatch.setattr(run_data, "_tokenizer", lambda p: calls.append(p) or original(p))
    path = Path(data["sources"][0]["prompt_data"])
    path.write_text(path.read_text().replace("a question", "changed question"))
    result = prepare(environment, data, tmp_path / "second")
    assert len(calls) == 1
    assert [r["status"] for r in json.loads(Path(result["manifest"]).read_text())["provenance"]["cache"]] == [
        "built",
        "hit",
    ]


def test_tokenizer_change_invalidates_both_sources(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    prepare(environment, data)
    (environment[0] / "tokenizer_config.json").write_text('{"chat_template":"changed"}')
    original = run_data._tokenizer
    calls = []
    monkeypatch.setattr(run_data, "_tokenizer", lambda p: calls.append(p) or original(p))
    prepare(environment, data, tmp_path / "second")
    assert len(calls) == 2


def test_corrupt_cache_and_import_fail(environment, tmp_path):
    data = config(tmp_path)
    first = prepare(environment, data)
    Path(first["train_sources"][0]).write_text("corrupt")
    with pytest.raises(InputError, match="artifact changed"):
        prepare(environment, data, tmp_path / "second")


def test_cross_source_holdout_overlap_still_rejected(environment, tmp_path, monkeypatch):
    data = config(tmp_path)
    prepare(environment, data)
    data["sources"][1]["eval_prompt_data"] = ["eval", data["sources"][0]["prompt_data"]]
    with pytest.raises(InputError, match="overlap"):
        prepare(environment, data, tmp_path / "second")


def test_schema_resolves_cache_paths_and_rejects_unsupported_source(tmp_path):
    data = config(tmp_path)
    data["preparation"] = {"cache_dir": "cache", "import_manifests": ["old/manifest.json"]}
    spec = test_miles_run_spec.spec(
        tmp_path,
        data=data,
        **{
            "async": {"fully_async": True, "prepared_batch_overlap": False},
            "records": {"enabled": True, "root": str(tmp_path / "records")},
        },
    )
    assert spec.data["preparation"]["cache_dir"] == str(tmp_path / "cache")
    assert spec.data["preparation"]["import_manifests"] == [str(tmp_path / "old/manifest.json")]
    data["sources"] = [{"name": "generated", "pool": "a", "tasks": [{"task": "multiplication", "train_count": 2}]}]
    with pytest.raises(InputError, match="rendered prompt_data"):
        preparation_cache.validate(data["preparation"], data)
