"""Publication gates cover the complete pool, not only successful sample rows."""

import asyncio
import json

import pytest
from scripts.data.rl_readiness import qualify_bundle

from open_instruct.miles.datasets import candidate_sources, run_data, source_catalog


class Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return list(text)


def setup_bundle(tmp_path, monkeypatch, extra=False):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_text('{"chat_template":"fixture"}')
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: Tokenizer())
    raw = tmp_path / "rows.jsonl"
    rows = [{"id": "one", "raw_record": {"prompt": "One plus one?", "target": {"kind": "numeric", "value": "2"}}}]
    if extra:
        rows.append(
            {"id": "two", "raw_record": {"prompt": "Return yes", "target": {"kind": "exact_text", "value": "yes"}}}
        )
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = tmp_path / "decon.json"
    receipt.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "toy": {
                        "adapter": "candidate_v1",
                        "inputs": [
                            {
                                "path": str(raw),
                                "sha256": candidate_sources.file_hash(raw),
                                "revision": "fixed",
                                "split": "train",
                            }
                        ],
                        "hygiene_receipt": {"path": str(receipt), "sha256": candidate_sources.file_hash(receipt)},
                    }
                },
            }
        )
    )
    candidate_sources.prepare(plan, tmp_path / "prepared", Tokenizer())
    controls = tmp_path / "controls.jsonl"
    controls.write_text(
        json.dumps(
            {
                "source": "toy",
                "sample_id": "one",
                "cases": [
                    {"kind": "gold", "response": "2", "expected": 1},
                    {"kind": "wrong", "response": "3", "expected": 0},
                    {"kind": "near_miss", "response": "12", "expected": 0},
                ],
            }
        )
        + "\n"
    )
    return tmp_path / "prepared/catalog.json", controls, hf


def test_complete_source_qualifies_and_resolves(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000))
    assert receipt["sources"]["toy"]["status"] == "qualified"
    assert receipt["sources"]["toy"]["controls"][0]["cases"][-1]["actual"] == 0
    data = source_catalog.resolve(tmp_path / "qualified/catalog.json", {"toy": 1.0})
    assert data["sources"][0]["name"] == "toy"
    (tmp_path / "qualified/qualification.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt hash"):
        source_catalog.resolve(tmp_path / "qualified/catalog.json", {"toy": 1.0})


def test_missing_stratum_never_shrinks_population(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch, extra=True)
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000))
    report = receipt["sources"]["toy"]
    assert report["status"] == "deferred" and report["missing_strata"] == ["candidate_v1:exact_text"]
    assert len((tmp_path / "qualified/toy/prompts.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="not qualified"):
        source_catalog.resolve(tmp_path / "qualified/catalog.json", {"toy": 1.0})


def test_whole_pool_length_validation(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 2))
    assert receipt["sources"]["toy"]["status"] == "deferred"
    assert "Prompt has" in receipt["sources"]["toy"]["reason"]


def test_named_judge_preserves_reference_and_requires_explicit_route():
    raw = {"prompt": "Prove the claim", "expected_answer": "Reference proof", "agent_ref": {"name": "math_with_judge"}}
    with pytest.raises(ValueError, match="explicit_adapter"):
        candidate_sources.adapt(raw, "nemo_v1")
    route = {"field": "agent_ref.name", "routes": {"math_with_judge": {"adapter": "reference_judge_v1"}}}
    messages, target, name, _ = candidate_sources.adapt(raw, "routed_v1", route)
    assert target == "Reference proof" and name == "candidate_reference"
    assert messages == [{"role": "user", "content": "Prove the claim"}]
    with pytest.raises(ValueError, match="unmapped"):
        candidate_sources.adapt(raw, "routed_v1", {**route, "routes": {}})
