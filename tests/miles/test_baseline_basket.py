"""Baseline holdout identity and physical allocation contracts."""

from pathlib import Path

import pytest

from scripts.miles import launch_baseline_basket, prepare_baseline_basket

from open_instruct.miles.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[2]


def row(verifier, index, *, identity=None):
    return {
        "input": f"{verifier}-{index}",
        "metadata": {"prepared_sample_id": identity or f"{verifier}-{index}", "verifiers": [{"name": verifier}]},
    }


def test_holdout_removes_duplicates_and_identity_aliases_preserves_train_multiplicity():
    names = ["math", "ifeval", "code", "general-quality"]
    old_eval = [row(name, 0) for name in names]
    train = [row(name, index) for name in names for index in range(1, 7)]
    train += [row("math", 0), row("math", "alias", identity="math-0")]
    train += [row("ifeval", 6)] * 3
    chosen = prepare_baseline_basket.select({"train": train, "eval": old_eval}, count=2)
    eval_rows = [r for key, rows in chosen.items() if key != "train" for r in rows]
    assert all(r in eval_rows for r in old_eval)
    prompts = {r["input"] for r in eval_rows}
    identities = {r["metadata"]["prepared_sample_id"] for r in eval_rows}
    assert all(
        r["input"] not in prompts and r["metadata"]["prepared_sample_id"] not in identities for r in chosen["train"]
    )
    expected = [
        r for r in train if r["input"] not in prompts and r["metadata"]["prepared_sample_id"] not in identities
    ]
    assert chosen["train"] == expected
    reversed_selection = prepare_baseline_basket.select({"train": list(reversed(train)), "eval": old_eval}, count=2)
    assert {key: rows for key, rows in chosen.items() if key != "train"} == {
        key: rows for key, rows in reversed_selection.items() if key != "train"
    }


def test_insufficient_domain_is_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_baseline_basket.select({"train": [row("math", 1)], "eval": []}, count=2)


def test_baseline_allocation_and_cpu_placement():
    run = RunSpec.load(ROOT / "configs/miles/qualification/full-sft-basket-fast-200.toml")
    layout = run.plan()["allocation"]
    assert (layout["replicas"], layout["allocated_gpus"], layout["unused_gpus"]) == (2, 16, 0)
    assert layout["nodes"][0]["trainer_gpus"] == 8
    assert layout["nodes"][1]["rollout_gpus"] == 7
    assert layout["nodes"][1]["judges"] == {"general": 1}
    compiled = run.compile()
    assert compiled.core.sequence_packing and not compiled.core.activation_checkpointing
    assert not compiled.core.scoring_pass_required
    assert compiled.miles["num_rollout"] == 200
    assert compiled.miles["global_batch_size"] == 256
    assert compiled.miles["n_samples_per_eval_prompt"] == 1
    spec = launch_baseline_basket.document("image", run, "prepare", "source", "a" * 64)
    assert len(spec["tasks"]) == 1
    task = spec["tasks"][0]
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}
    assert "gpuCount" not in task["resources"]
    assert "scripts.miles.prepare_baseline_basket" in task["arguments"][0]
    assert "open_instruct.miles.cluster /output" not in task["arguments"][0]
