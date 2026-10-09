"""Evaluation panels preserve identities, response caps and final full coverage."""

import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles.rollout import bread_evaluation, bread_observation


@dataclasses.dataclass
class Dataset:
    name: str
    path: str
    n_samples_per_eval_prompt: int = 2
    max_response_len: int = 16384


def test_panel_initial_long_periodic_and_final(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps({"id": i}) for i in range(4)) + "\n")
    args = SimpleNamespace(
        save=str(tmp_path),
        eval_datasets=[Dataset("heldout", str(path))],
        eval_reward_key=None,
        reward_key="score",
        olmo_core=SimpleNamespace(bread_eval_panel_size=2, bread_eval_long_response_len=32768),
    )
    calls = []

    async def native(config):
        ids = [json.loads(line)["id"] for line in Path(config.path).read_text().splitlines()]
        calls.append((config.name, ids, config.max_response_len))
        samples = [
            SimpleNamespace(
                index=i * 2 + j,
                metadata={"prepared_sample_id": str(identity)},
                status="completed" if j == 0 else "truncated",
                response="</think> answer" if j == 0 else "thinking",
                response_length=10,
                reward={"score": 1 - j},
            )
            for i, identity in enumerate(ids)
            for j in range(2)
        ]
        return {
            config.name: {
                "samples": samples,
                "rewards": [s.reward["score"] for s in samples],
                "truncated": [s.status == "truncated" for s in samples],
            }
        }

    initial, metrics = asyncio.run(
        bread_evaluation.evaluate(args, SimpleNamespace(rollout_id=0), native, initial=True)
    )
    assert calls == [("heldout_full", [0, 1, 2, 3], 16384), ("heldout_panel_long", [0, 1], 32768)]
    assert len(initial["heldout_panel"]["samples"]) == 4
    assert metrics["eval/heldout_panel/accuracy_finished"] == 1
    assert initial["heldout_panel"]["rewards"] == [1, 0, 1, 0]
    calls.clear()
    periodic, _ = asyncio.run(bread_evaluation.evaluate(args, SimpleNamespace(rollout_id=4), native, initial=False))
    assert set(periodic) == {"heldout_panel"}
    assert calls == [("heldout_panel", [0, 1], 16384)]
    bread_evaluation.mark_final(args, 7)
    final, _ = asyncio.run(bread_evaluation.evaluate(args, SimpleNamespace(rollout_id=7), native, initial=False))
    assert len(final["heldout_full"]["samples"]) == 8
    records = [json.loads(line) for line in (tmp_path / "bread/eval-responses.jsonl").read_text().splitlines()]
    assert any(r["final"] and r["rollout_id"] == 7 for r in records)


def test_generation_timing_separates_admission_and_service(monkeypatch):
    times = iter([13.0, 18.0])
    monkeypatch.setattr(bread_observation.time, "time", lambda: next(times))
    sample = SimpleNamespace(metadata={"bread_request_submitted_unix": 10.0})

    async def generate(input):
        return SimpleNamespace(samples=input)

    asyncio.run(bread_observation.timed_generator(generate)(sample))
    assert sample.metadata["bread_timing"]["client_admission_wait_seconds"] == 3
    assert sample.metadata["bread_timing"]["generation_call_seconds"] == 5


def test_long_evaluation_context_does_not_change_training_state():
    state = SimpleNamespace(args=SimpleNamespace(sglang_context_length=65536, rollout_max_context_len=32768))
    evaluation = bread_evaluation.evaluation_state(state)
    assert evaluation.args.rollout_max_context_len == 65536
    assert state.args.rollout_max_context_len == 32768
