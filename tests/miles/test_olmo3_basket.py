"""Keep checkpoint comparisons fixed at the prompt, split and verifier boundary."""

import hashlib
import json
from pathlib import Path

import pytest
from scripts.miles import launch_baseline_basket, prepare_olmo3_basket

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/miles/examples/medium.toml"


def test_dense_basket_configuration():
    run = RunSpec.load(
        CONFIG,
        overrides=[
            "trainer.expert_parallel_size=1",
            "miles.use_rollout_routing_replay=false",
            "inference.radix_cache=false",
            "inference.sglang_max_total_tokens=131072",
        ],
    )
    compiled = run.compile()
    assert compiled.core.expert_parallel_size == 1
    assert compiled.core.sequence_packing
    assert not compiled.miles["use_rollout_routing_replay"]
    assert compiled.miles["num_rollout"] == 200
    assert compiled.miles["global_batch_size"] == 256
    assert compiled.miles["sglang_max_total_tokens"] == 131072
    assert run.plan()["allocation"]["allocated_gpus"] == 16
    document = launch_baseline_basket.document(
        "image", run, "prepare", "source", "a" * 64, prepare_module="scripts.miles.prepare_olmo3_basket"
    )
    (task,) = document["tasks"]
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}
    assert task["resources"]["gpuCount"] == 0
    assert "python -m scripts.miles.prepare_olmo3_basket" in task["arguments"][0]


@pytest.mark.parametrize("failure", [None, "hash", "length", "overlap"])
def test_frozen_retokenization(tmp_path, monkeypatch, failure):
    run = RunSpec.load(
        CONFIG,
        overrides=[
            "trainer.expert_parallel_size=1",
            "miles.use_rollout_routing_replay=false",
            "inference.radix_cache=false",
            "inference.sglang_max_total_tokens=131072",
        ],
    )
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "target"
    run.data["prompt_data"] = str(output / "train.jsonl")
    row = {
        "input": "literal prompt",
        "label": "answer",
        "metadata": {
            "prepared_sample_id": "train",
            "verifiers": [{"name": "math", "target": "answer"}],
            "run_prompt_token_ids_sha256": "old-tokenizer-hash",
        },
    }
    train = [row, row]  # Preserve source multiplicity and order.
    evaluation = json.loads(json.dumps(row))
    if failure != "overlap":
        evaluation["input"] = "held out prompt"
        evaluation["metadata"]["prepared_sample_id"] = "eval"
    for name, rows in [("train", train), ("eval-math", [evaluation])]:
        (source / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (source / "verifiers.json").write_text(json.dumps({"math": {"factory": "x.y"}}))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    if failure == "hash":
        (source / "train.jsonl").write_text("changed")
    tokenizer = type(
        "Tokenizer", (), {"encode": lambda self, text, **kw: list(range(3000 if failure == "length" else 3))}
    )()

    async def canaries():
        return [1, 0]

    monkeypatch.setattr(prepare_olmo3_basket.prepare_baseline_basket, "code_canaries", canaries)
    write = prepare_olmo3_basket.workflow.write_json
    monkeypatch.setattr(
        prepare_olmo3_basket.workflow,
        "write_json",
        lambda path, value: write(
            tmp_path / "receipt.json" if str(path) == "/output/preparation.json" else path, value
        ),
    )
    if failure:
        with pytest.raises((ValueError, InputError)):
            prepare_olmo3_basket.prepare_data(run, tokenizer, source, hashes)
        assert not output.exists()
        return
    prepare_olmo3_basket.prepare_data(run, tokenizer, source, hashes)
    actual = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(actual) == 2
    assert all(r["input"] == row["input"] and r["label"] == row["label"] for r in actual)
    assert all(r["metadata"]["verifiers"] == row["metadata"]["verifiers"] for r in actual)
    report = json.loads((output / "preparation.json").read_text())
    assert report["held_out_ids"] == {"eval-math": ["eval"]}
    assert report["changed_prompt_tokenizations"]["train"] == 2
    prepare_olmo3_basket.prepare_baseline_basket.verify(run)
