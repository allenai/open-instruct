"""Publication gates cover the complete pool, not only successful sample rows."""

import asyncio
import json
import shutil

import pytest
from scripts.data.rl_readiness import qualify_bundle

from open_instruct.miles.datasets import run_data, source_catalog, source_preparation


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
                        "adapter": "deterministic_v1",
                        "inputs": [
                            {
                                "path": str(raw),
                                "sha256": source_preparation.file_hash(raw),
                                "revision": "fixed",
                                "split": "train",
                            }
                        ],
                        "hygiene_receipt": {"path": str(receipt), "sha256": source_preparation.file_hash(receipt)},
                    }
                },
            }
        )
    )
    source_preparation.prepare(plan, tmp_path / "prepared", Tokenizer())
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
    # Existing catalogs use a human-readable qualification annotation.
    legacy = json.loads((tmp_path / "qualified/catalog.json").read_text())
    legacy["sources"]["toy"]["qualification"] = "Native controls and loader passed"
    (tmp_path / "qualified/legacy.json").write_text(json.dumps(legacy))
    assert source_catalog.resolve(tmp_path / "qualified/legacy.json", {"toy": 1.0})["sources"][0]["name"] == "toy"
    (tmp_path / "qualified/qualification.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt hash"):
        source_catalog.resolve(tmp_path / "qualified/catalog.json", {"toy": 1.0})


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_qualification_preserves_unicode_line_separators(tmp_path, monkeypatch, separator):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    document = json.loads(catalog.read_text())
    entry = document["sources"]["toy"]
    prompts = catalog.parent / entry["prompt_data"]["path"]
    row = json.loads(prompts.read_text())
    row["input"] = "Question" + separator + "One plus one?"
    prompts.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    entry["prompt_data"]["sha256"] = source_preparation.file_hash(prompts)
    catalog.write_text(json.dumps(document))
    control = json.loads(controls.read_text())
    control["provenance"] = "Line one" + separator + "line two"
    controls.write_text(json.dumps(control, ensure_ascii=False) + "\n")
    output = tmp_path / "qualified"
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, output, 1000))
    assert receipt["sources"]["toy"]["status"] == "qualified"
    assert json.loads((output / "toy/prompts.jsonl").read_text())["input"] == row["input"]


def test_missing_stratum_never_shrinks_population(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch, extra=True)
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000))
    report = receipt["sources"]["toy"]
    assert report["status"] == "deferred" and report["missing_strata"] == ["deterministic_v1:exact_text"]
    assert len((tmp_path / "qualified/toy/prompts.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="not qualified"):
        source_catalog.resolve(tmp_path / "qualified/catalog.json", {"toy": 1.0})


@pytest.mark.parametrize("artifact", ["prompt_data", "reward_config"])
def test_receipt_rejects_changed_artifact_even_with_updated_catalog(tmp_path, monkeypatch, artifact):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    output = tmp_path / "qualified"
    asyncio.run(qualify_bundle.qualify(catalog, controls, hf, output, 1000))
    published = output / "catalog.json"
    document = json.loads(published.read_text())
    descriptor = document["sources"]["toy"][artifact]
    path = output / descriptor["path"]
    value = json.loads(path.read_text())
    if artifact == "prompt_data":
        value["input"] = "A different question"
    else:
        value["deterministic"]["factory"] = source_preparation.FACTORIES["math"]
    path.write_text(json.dumps(value) + "\n")
    descriptor["sha256"] = source_preparation.file_hash(path)
    published.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=f"qualification receipt {artifact} hash mismatch"):
        source_catalog.resolve(published, {"toy": 1.0})


@pytest.mark.parametrize("missing", ["artifact_sha256", "prompt_data", "reward_config"])
def test_receipt_requires_both_artifact_bindings(tmp_path, monkeypatch, missing):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    output = tmp_path / "qualified"
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, output, 1000))
    report = receipt["sources"]["toy"]
    (report if missing == "artifact_sha256" else report["artifact_sha256"]).pop(missing)
    receipt_path = output / "qualification.json"
    receipt_path.write_text(json.dumps(receipt))
    published = output / "catalog.json"
    document = json.loads(published.read_text())
    document["sources"]["toy"]["qualification"]["sha256"] = source_preparation.file_hash(receipt_path)
    published.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="requalify the bundle"):
        source_catalog.resolve(published, {"toy": 1.0})


def test_qualification_survives_relocation_and_can_be_renewed(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    output = tmp_path / "qualified"
    asyncio.run(qualify_bundle.qualify(catalog, controls, hf, output, 1000))
    relocated = tmp_path / "relocated"
    shutil.copytree(output, relocated)
    published = relocated / "catalog.json"
    resolved = source_catalog.resolve(published, {"toy": 1.0})
    assert resolved["sources"][0]["prompt_data"] == str(relocated / "toy/prompts.jsonl")
    document = json.loads(published.read_text())
    descriptor = document["sources"]["toy"]["prompt_data"]
    path = relocated / descriptor["path"]
    row = json.loads(path.read_text())
    row["input"] = "What is 1 + 1?"
    path.write_text(json.dumps(row) + "\n")
    descriptor["sha256"] = source_preparation.file_hash(path)
    published.write_text(json.dumps(document))
    renewed = tmp_path / "requalified"
    receipt = asyncio.run(qualify_bundle.qualify(published, controls, hf, renewed, 1000))
    assert receipt["sources"]["toy"]["artifact_sha256"]["prompt_data"] == descriptor["sha256"]
    assert source_catalog.resolve(renewed / "catalog.json", {"toy": 1.0})["sources"][0]["name"] == "toy"


def test_whole_pool_length_validation(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 2))
    assert receipt["sources"]["toy"]["status"] == "deferred"
    assert "Prompt has" in receipt["sources"]["toy"]["reason"]


def test_quarantining_every_row_cannot_qualify_an_empty_source(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    document = json.loads(catalog.read_text())
    entry = document["sources"]["toy"]
    prompts = catalog.parent / entry["prompt_data"]["path"]
    prompts.write_text("")
    entry["prompt_data"]["sha256"] = source_preparation.file_hash(prompts)
    entry["counts"].update(prepared_rows=0, omitted_rows=1)
    dispositions = catalog.parent / entry["dispositions"]["path"]
    dispositions.write_text(json.dumps({"id": "one", "status": "omitted", "reason": "Malformed reference"}) + "\n")
    entry["dispositions"]["sha256"] = source_preparation.file_hash(dispositions)
    catalog.write_text(json.dumps(document))
    controls.write_text("")
    quarantine = tmp_path / "quarantine.jsonl"
    quarantine.write_text(json.dumps({"source": "toy", "id": "one", "reason": "Malformed reference"}) + "\n")
    output = tmp_path / "qualified"
    receipt = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, output, 1000, quarantine))
    assert receipt["sources"]["toy"]["status"] == "deferred"
    with pytest.raises(ValueError, match="not qualified"):
        source_catalog.resolve(output / "catalog.json", {"toy": 1.0})


def test_named_judge_preserves_reference_and_requires_explicit_route():
    raw = {"prompt": "Prove the claim", "expected_answer": "Reference proof", "agent_ref": {"name": "math_with_judge"}}
    with pytest.raises(ValueError, match="explicit_adapter"):
        source_preparation.adapt(raw, "nemo_v1")
    route = {"field": "agent_ref.name", "routes": {"math_with_judge": {"adapter": "reference_judge_v1"}}}
    messages, target, name, _ = source_preparation.adapt(raw, "routed_v1", route)
    assert target == "Reference proof" and name == "reference_judge"
    assert messages == [{"role": "user", "content": "Prove the claim"}]
    with pytest.raises(ValueError, match="unmapped"):
        source_preparation.adapt(raw, "routed_v1", {**route, "routes": {}})


@pytest.mark.parametrize("accepted", [False, True])
def test_quarantine_requires_exact_omission_identities(tmp_path, monkeypatch, accepted):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    document = json.loads(catalog.read_text())
    entry = document["sources"]["toy"]
    entry["counts"]["omitted_rows"] = 1
    dispositions = catalog.parent / entry["dispositions"]["path"]
    dispositions.write_text(
        json.dumps({"id": "bad-row", "status": "omitted", "reason": "Invalid source schema"}) + "\n"
    )
    entry["dispositions"]["sha256"] = source_preparation.file_hash(dispositions)
    catalog.write_text(json.dumps(document))
    quarantine = tmp_path / "reviewed.jsonl"
    quarantine.write_text(
        json.dumps(
            {
                "source": "toy",
                "id": "bad-row" if accepted else "another-row",
                "reason": "Malformed source; quarantined without repair",
            }
        )
        + "\n"
    )
    report = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000, quarantine))
    assert report["sources"]["toy"]["status"] == ("qualified" if accepted else "deferred")
    assert (tmp_path / "qualified/quarantine.jsonl").read_text() == quarantine.read_text()
    assert (tmp_path / "qualified/toy/dispositions.jsonl").read_text() == dispositions.read_text()


def test_timeout_cannot_pass_a_zero_reward_control(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)

    async def reward(args, sample):
        if sample.response == "2":
            return 1.0
        sample.metadata["verifier_diagnostics"] = {"test": {"status": "timeout"}}
        return 0.0

    monkeypatch.setattr(qualify_bundle.rewards, "registered_reward", reward)
    report = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000))
    assert report["sources"]["toy"]["status"] == "deferred"
    assert "wrong control timed out" in report["sources"]["toy"]["reason"]


def test_native_partial_credit_is_an_explicit_control_expectation(tmp_path, monkeypatch):
    catalog, controls, hf = setup_bundle(tmp_path, monkeypatch)
    value = json.loads(controls.read_text())
    value["cases"][2]["expected"] = 0.5
    controls.write_text(json.dumps(value) + "\n")

    async def reward(args, sample):
        if sample.status == "truncated":
            return 0.0
        return {"2": 1.0, "3": 0.0, "12": 0.5, "": 0.0}[sample.response]

    monkeypatch.setattr(qualify_bundle.rewards, "registered_reward", reward)
    report = asyncio.run(qualify_bundle.qualify(catalog, controls, hf, tmp_path / "qualified", 1000))
    assert report["sources"]["toy"]["status"] == "qualified"


def test_different_code_test_suites_require_a_shared_passing_witness():
    rows = [
        {"input": "Implement f", "metadata": {"sample_id": str(i), "verifiers": [{"name": "code", "target": [test]}]}}
        for i, test in enumerate(["assert f(1)==1", "assert f(2)==2"])
    ]
    registry = {"code": {"factory": source_preparation.FACTORIES["complete_code"]}}
    controls = [
        {"sample_id": str(i), "cases": [{"kind": "gold", "response": "def f(x): return x", "expected": 1}]}
        for i in range(2)
    ]
    qualify_bundle.validate_prompt_targets(rows, controls, registry)
    with pytest.raises(ValueError, match="shared gold"):
        qualify_bundle.validate_prompt_targets(rows, controls[:1], registry)
    registry["code"]["factory"] = source_preparation.FACTORIES["calendar"]
    with pytest.raises(ValueError, match="Conflicting targets"):
        qualify_bundle.validate_prompt_targets(rows, controls, registry)


@pytest.mark.parametrize("has_calls", [True, False])
def test_toolrl_qualification_preserves_native_reward_scale(tmp_path, has_calls):
    target = {
        "format": "olmo-toolrl-v1",
        "tool_format": "olmo_pythonic_v1",
        "think_prefix_in_prompt": True,
        "tools": [{"function": {"name": "lookup", "parameters": {"properties": {"item": {"type": "string"}}}}}],
        "calls": [{"name": "lookup", "parameters": {"item": "apple"}}] if has_calls else [],
    }
    row = {"input": "<think>", "metadata": {"verifiers": [{"name": "toolrl", "target": target}]}}
    reward_path = tmp_path / "rewards.json"
    reward_path.write_text(json.dumps({"toolrl": {"factory": source_preparation.FACTORIES["toolrl"]}}))
    gold = 'plan</think><function_calls>lookup(item="apple")</function_calls>' if has_calls else "plan</think>answer"
    near = gold.replace("apple", "pear") if has_calls else "plan</think>"
    cases = [
        {"kind": "gold", "response": gold, "expected": 4 if has_calls else 1},
        {"kind": "wrong", "response": "invalid", "expected": -3 if has_calls else 0},
        {"kind": "near_miss", "response": near, "expected": 2 if has_calls else 0},
    ]
    results = asyncio.run(qualify_bundle.check_control(row, cases, reward_path))
    assert [r["actual"] for r in results] == ([4, -3, 2, -3, 0] if has_calls else [1, 0, 0, 0, 0])
    cases[0]["expected"] = 1 if has_calls else 4
    with pytest.raises(ValueError, match="Gold must"):
        asyncio.run(qualify_bundle.check_control(row, cases, reward_path))


def test_instruction_family_projection_keeps_complete_target_unchanged():
    target = {
        "family": "nemo",
        "target": {
            "instruction_id_list": ["first", "second"],
            "kwargs": [{"x": 1}, {"y": 2}],
            "grading_mode": "fraction",
        },
    }
    row = {"input": "original prompt", "label": target, "metadata": {"verifiers": [{"name": "if", "target": target}]}}
    registry = {"if": {"factory": source_preparation.FACTORIES["instruction"]}}
    projected, family = qualify_bundle.family_control_row(row, 1, registry)
    assert family == "second"
    assert projected["label"]["target"] == {
        "instruction_id_list": ["second"],
        "kwargs": [{"y": 2}],
        "grading_mode": "fraction",
    }
    assert row["label"]["target"]["instruction_id_list"] == ["first", "second"]
    with pytest.raises(ValueError, match="constraint_index"):
        qualify_bundle.family_control_row(row, -1, registry)
    with pytest.raises(ValueError, match="only supported"):
        qualify_bundle.instruction_families(row, {"if": {"factory": source_preparation.FACTORIES["math"]}})


def test_first_pass_family_coverage_requires_every_family_and_keeps_bundles(tmp_path, monkeypatch):
    catalog_path, controls_path, hf = setup_bundle(tmp_path, monkeypatch)
    catalog = json.loads(catalog_path.read_text())
    entry = catalog["sources"]["toy"]
    prompt_path = catalog_path.parent / entry["prompt_data"]["path"]
    registry_path = catalog_path.parent / entry["reward_config"]["path"]
    row = json.loads(prompt_path.read_text())
    target = {"instruction_id": ["detectable_format:title", "startend:quotation"], "kwargs": [{}, {}]}
    row["label"] = target
    row["metadata"]["verifiers"] = [{"name": "ifeval", "target": target}]
    prompt_path.write_text(json.dumps(row) + "\n")
    registry_path.write_text(json.dumps({"ifeval": {"factory": source_preparation.FACTORIES["ifeval"]}}))
    for key, path in [("prompt_data", prompt_path), ("reward_config", registry_path)]:
        entry[key]["sha256"] = source_preparation.file_hash(path)
    catalog_path.write_text(json.dumps(catalog))
    controls = [
        {
            "source": "toy",
            "sample_id": "one",
            "constraint_index": i,
            "cases": [
                {"kind": "gold", "response": gold, "expected": 1},
                {"kind": "wrong", "response": "unformatted", "expected": 0},
                {"kind": "near_miss", "response": near, "expected": 0},
            ],
        }
        for i, (gold, near) in enumerate([("<<Title>>", "<<Title>"), ('"answer"', '"answer')])
    ]
    controls_path.write_text(json.dumps(controls[0]) + "\n")
    missing = asyncio.run(
        qualify_bundle.qualify(
            catalog_path, controls_path, hf, tmp_path / "missing-family", 1000, family_coverage=True
        )
    )
    assert missing["sources"]["toy"]["status"] == "deferred"
    assert missing["sources"]["toy"]["missing_strata"] == ["startend:quotation"]
    controls_path.write_text("".join(json.dumps(c) + "\n" for c in controls))
    passed = asyncio.run(
        qualify_bundle.qualify(catalog_path, controls_path, hf, tmp_path / "all-families", 1000, family_coverage=True)
    )
    assert passed["sources"]["toy"]["status"] == "qualified"
    assert passed["sources"]["toy"]["coverage"] == "instruction_families"
    assert json.loads((tmp_path / "all-families/toy/prompts.jsonl").read_text())["label"] == target


def test_shared_citation_witness_requires_every_target():
    rows = [
        {
            "input": "same prompt",
            "metadata": {
                "sample_id": str(i),
                "verifiers": [
                    {
                        "name": "format",
                        "target": {"kind": "format", "value": {"type": "string_match", "expected_markers": markers}},
                    }
                ],
            },
        }
        for i, markers in enumerate([["[ref:1]"], ["[ref:1]", "[ref:2]"]])
    ]
    registry = {"format": {"factory": source_preparation.FACTORIES["deterministic"]}}
    controls = [
        {"sample_id": str(i), "cases": [{"kind": "gold", "response": "[ref:1] [ref:2]", "expected": 1}]}
        for i in range(2)
    ]
    qualify_bundle.validate_prompt_targets(rows, controls, registry)
    with pytest.raises(ValueError, match="shared gold"):
        qualify_bundle.validate_prompt_targets(rows, controls[:1], registry)


def test_equivalent_math_targets_require_shared_runtime_witness(tmp_path):
    rows = [
        {
            "input": "What is 52 divided by 3?",
            "metadata": {"sample_id": str(i), "verifiers": [{"name": "math", "target": target}]},
        }
        for i, target in enumerate(["52 / 3", "52/3"])
    ]
    registry = {"math": {"factory": source_preparation.FACTORIES["math"]}}
    controls = [
        {
            "sample_id": str(i),
            "cases": [
                {"kind": "gold", "response": r"\boxed{52/3}", "expected": 1},
                {"kind": "wrong", "response": r"\boxed{17}", "expected": 0},
                {"kind": "near_miss", "response": r"\boxed{53/3}", "expected": 0},
            ],
        }
        for i in range(2)
    ]
    qualify_bundle.validate_prompt_targets(rows, controls, registry)
    with pytest.raises(ValueError, match="shared gold"):
        qualify_bundle.validate_prompt_targets(rows, controls[:1], registry)
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(registry))
    for row, control in zip(rows, controls):
        asyncio.run(qualify_bundle.check_control(row, control["cases"], path))
    rows[1]["metadata"]["verifiers"][0]["target"] = "19"
    with pytest.raises(ValueError, match="gold expected 1"):
        asyncio.run(qualify_bundle.check_control(rows[1], controls[1]["cases"], path))


def test_reasoning_gym_duplicate_uuid_is_provenance_only():
    rows = [
        {
            "input": "same puzzle",
            "metadata": {
                "sample_id": str(i),
                "verifiers": [
                    {
                        "name": "rg",
                        "target": {"uuid": str(i), "answer": "bob", "metadata": {"source_dataset": "zebra_puzzles"}},
                    }
                ],
            },
        }
        for i in range(2)
    ]
    registry = {"rg": {"factory": source_preparation.FACTORIES["reasoning_gym"]}}
    qualify_bundle.validate_prompt_targets(rows, [], registry)
    assert rows[0]["metadata"]["verifiers"][0]["target"]["uuid"] == "0"
    rows[1]["metadata"]["verifiers"][0]["target"]["answer"] = "alice"
    with pytest.raises(ValueError, match="Conflicting targets"):
        qualify_bundle.validate_prompt_targets(rows, [], registry)


@pytest.mark.parametrize("status", ["judge_error", "service_error"])
@pytest.mark.parametrize("failed_case", ["wrong", "near_miss"])
def test_failed_negative_grades_cannot_qualify_after_successful_gold(tmp_path, monkeypatch, status, failed_case):
    reward_path = tmp_path / "rewards.json"
    reward_path.write_text(json.dumps({"judge": {"factory": source_preparation.FACTORIES["reference_judge"]}}))
    row = {"input": "question", "metadata": {"verifiers": [{"name": "judge", "target": "answer"}]}}
    cases = [
        {"kind": kind, "response": kind, "expected": int(kind == "gold")} for kind in ["gold", "wrong", "near_miss"]
    ]
    seen = []

    async def grade(args, sample):
        seen.append(sample.response)
        if sample.response == failed_case:
            sample.metadata["verifier_diagnostics"] = {"judge": {"status": status, "error": "backend unavailable"}}
        return float(sample.response == "gold")

    monkeypatch.setattr(qualify_bundle.rewards, "registered_reward", grade)
    with pytest.raises(ValueError, match=f"{failed_case} control was not graded: {status}"):
        asyncio.run(qualify_bundle.check_control(row, cases, reward_path))
    assert seen == (["gold", "wrong"] if failed_case == "wrong" else ["gold", "wrong", "near_miss"])
