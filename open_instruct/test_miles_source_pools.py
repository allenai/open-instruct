"""Source-level pool assignment uses the same finite sampler as row labels."""

import copy
import json
from pathlib import Path

import pytest

from open_instruct import test_miles_run_data, test_miles_run_spec
from open_instruct.miles.datasets import pools, run_data

environment = test_miles_run_data.environment


def sources():
    return [
        {"name": name, "pool": name, "prompt_data": f"{name}.jsonl", "reward_config": "rewards.json"}
        for name in ("deepmath", "polaris")
    ]


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
def test_source_pool_compiles_without_row_label_configuration(tmp_path, policy):
    data = {"sources": sources(), "sampling": {"policy": policy}}
    if policy == "weighted":
        data["sampling"]["weights"] = {"deepmath": 3, "polaris": 1}
    spec = test_miles_run_spec.spec(
        tmp_path,
        data=data,
        records={"enabled": True, "root": "/records"},
        core={"prepared_batch_overlap": False, "prompt_replay_enabled": True},
        **{"async": {"fully_async": True}},
    )
    assert "pools" not in spec.data
    assert [s["pool"] for s in spec.data["sources"]] == ["deepmath", "polaris"]
    assert spec.data["sources"][0]["prompt_data"] == str(tmp_path / "deepmath.jsonl")
    compiled = spec.compile()
    assert compiled.core.prompt_provider == "pools" and compiled.core.prompt_replay_enabled
    assert compiled.miles["async_unused_samples_handler"] == "drop"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("existing_label", [None, "original-topic"])
def test_preparation_assigns_sources_preserves_payload_and_pins_membership(
    environment, tmp_path, shared, existing_label
):
    config = sources()
    original = {}
    for source in config:
        if shared:
            source["pool"] = "math"
        row = {
            "input": f"Question for {source['name']}",
            "label": "42",
            "metadata": {"prepared_sample_id": source["name"], "verifiers": [{"name": "gsm8k", "target": "42"}]},
        }
        if existing_label:
            row["pool_id"] = existing_label
        original[source["name"]] = copy.deepcopy(row)
        path = tmp_path / source["prompt_data"]
        path.write_text(json.dumps(row) + "\n")
        source["prompt_data"] = str(path)
        source["reward_config"] = str(tmp_path / "rewards.json")
    rewards = {"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}
    (tmp_path / "rewards.json").write_text(json.dumps(rewards))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps(
            {
                "input": "heldout question",
                "label": "42",
                "metadata": original["deepmath"]["metadata"] | {"prepared_sample_id": "eval"},
            }
        )
    )
    config[0]["eval_prompt_data"] = ["test", str(heldout)]
    data = {"sources": config, "sampling": {"policy": "uniform"}}
    result = test_miles_run_data.prepare(environment, data)
    manifest = json.loads(Path(result["prompt_mixture"]).read_text())
    assert manifest["pools"] == {s["pool"]: {"label": s["pool"]} for s in config}
    assert manifest["sampling"] == {"policy": "uniform"}
    for source, prepared in zip(config, manifest["sources"], strict=True):
        row = json.loads((Path(result["prompt_mixture"]).parent / prepared["path"]).read_text())
        assert prepared["pool"] == row["pool_id"] == row["metadata"]["sampling_pool"] == source["pool"]
        assert row["metadata"]["verifiers"] == original[source["name"]]["metadata"]["verifiers"]
        assert row["input"] == original[source["name"]]["input"] and row["label"] == "42"
        assert json.loads(Path(source["prompt_data"]).read_text()) == original[source["name"]]
    assert "pool_id" not in json.loads(Path(result["eval_prompt_data"][1]).read_text())
    assert test_miles_run_data.prepare(environment, data) == result
    changed = copy.deepcopy(data)
    changed["sources"][0]["pool"] = "new-pool"
    with pytest.raises(ValueError, match="contract"):
        test_miles_run_data.prepare(environment, changed)


def test_source_assignment_and_row_label_assignment_can_share_pool():
    data = {"sources": sources(), "pools": {"math": {"label": "mathematics"}}}
    data["sources"][0]["pool"] = "math"
    data["sources"][1].pop("pool")
    run_data.validate_data(data)
    resolved = pools.resolve(data)
    assigned = {"metadata": {}, "pool_id": "old-topic"}
    labeled = {"metadata": {}, "pool_id": "mathematics"}
    pools.annotate(assigned, resolved, source_pool="math")
    pools.annotate(labeled, resolved)
    assert assigned == labeled
    assert resolved == data["pools"]


@pytest.mark.parametrize("value", [None, "", True, 3, {}, "../unsafe", "has space"])
def test_invalid_source_pool_rejected(value):
    data = {"sources": sources()}
    data["sources"][0]["pool"] = value
    with pytest.raises(ValueError, match="Source pool"):
        run_data.validate_data(data)


def test_partial_assignment_and_fraction_conflicts_are_rejected():
    data = {"sources": sources()}
    data["sources"][0].pop("pool")
    with pytest.raises(ValueError, match="every source"):
        run_data.validate_data(data)
    data["sources"][0]["pool"] = "deepmath"
    data["sources"][0]["fraction"] = 0.5
    with pytest.raises(ValueError, match="omit source fraction"):
        run_data.validate_data(data)


def test_resolving_pools_does_not_mutate_configuration():
    data = {"sources": sources(), "pools": {"math": {"label": "mathematics"}}}
    original = copy.deepcopy(data)
    assert set(pools.resolve(data)) == {"math", "deepmath", "polaris"}
    assert data == original


def test_generated_task_source_uses_same_assignment(environment):
    data = {
        "sources": [{"name": "arithmetic", "pool": "math", "tasks": [{"task": "multiplication", "train_count": 3}]}]
    }
    result = test_miles_run_data.prepare(environment, data)
    rows = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["pool_id"] == row["metadata"]["sampling_pool"] == "math" for row in rows)
