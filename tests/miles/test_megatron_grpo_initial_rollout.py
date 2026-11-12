"""Megatron verifier GRPO continuing another trainer's checkpoint (training.initial_rollout)."""

import json
from pathlib import Path

import pytest
import tomllib
import torch
from safetensors import torch as safetensors_torch

from open_instruct.miles.configuration import specs
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import workflow
from open_instruct.miles.megatron_grpo import grpo_args, grpo_audit, grpo_hooks

FIXTURE = Path(__file__).parent / "fixtures/megatron_grpo/qwen35-2b-verifier-dppo-async-short.toml"


def load(root=None, **training):
    document = tomllib.loads(FIXTURE.read_text())
    document["training"].update({"resume": True} | training)
    if root is not None:
        document["output"] = {"root": str(root / "run"), "assets": str(root / "assets")}
    return document


def seed(spec, iteration):
    marker = Path(spec.output["root"]) / "checkpoints/latest_checkpointed_iteration.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(iteration))


def test_initial_rollout_defaults_off_without_changing_existing_run_identity():
    document = specs.from_dict(load(), config_path=FIXTURE).to_dict()
    assert document["training"]["initial_rollout"] == 0
    recorded = {**document, "training": {k: v for k, v in document["training"].items() if k != "initial_rollout"}}
    assert workflow.fingerprint(document) == workflow.fingerprint(recorded)
    continued = specs.from_dict(load(initial_rollout=6), config_path=FIXTURE).to_dict()
    assert workflow.fingerprint(continued) != workflow.fingerprint(document)


@pytest.mark.parametrize(
    "training, message",
    [
        ({"initial_rollout": -1}, "training.initial_rollout"),
        ({"initial_rollout": 10}, "below training.num_rollouts"),
        ({"initial_rollout": 6, "resume": False}, "requires training.resume"),
    ],
)
def test_initial_rollout_rejects_invalid_values(training, message):
    with pytest.raises(InputError, match=message):
        specs.from_dict(load(**training), config_path=FIXTURE)


def test_unseeded_root_refuses_a_fresh_start(tmp_path):
    spec = specs.from_dict(load(tmp_path, initial_rollout=6), config_path=FIXTURE)
    with pytest.raises(InputError, match="refusing a fresh start"):
        grpo_args.require_seeded_state(spec)
    seed(spec, 4)  # the seeding trainer's checkpoint is one rollout short
    with pytest.raises(InputError, match="iteration 5"):
        grpo_args.require_seeded_state(spec)
    seed(spec, 5)
    grpo_args.require_seeded_state(spec)
    # The seeded checkpoint resumes like any GRPO checkpoint: weights, optimizer, RNG and scheduler.
    prepared = {"model": "/m", "data": {"prompt_data": "/p", "eval_prompt_data": ["e", "/e"]}}
    args = grpo_args.native_arguments(spec, prepared, "/converted", [])
    assert args[args.index("--load") + 1] == str(Path(spec.output["root"]) / "checkpoints")
    assert "--use-checkpoint-opt-param-scheduler" in args
    assert args[args.index("--num-rollout") + 1] == "10"
    assert grpo_args.checkpointed_rollouts(spec) == 6


def test_reconcile_counts_only_this_trainers_rollouts(tmp_path):
    # Seeded at rollout 6, then rollouts 6 and 7 trained and checkpointed, then a stop during rollout 8.
    batch = 4
    lines = [json.dumps({"sample_index": index}) + "\n" for index in range(3 * batch)]
    (tmp_path / grpo_hooks.EVIDENCE).write_text("".join(lines))
    assert grpo_hooks.reconcile_evidence(tmp_path, 8 - 6, batch) == batch
    assert (tmp_path / grpo_hooks.EVIDENCE).read_text() == "".join(lines[: 2 * batch])


def test_audit_covers_only_rollouts_after_the_handoff(tmp_path, monkeypatch):
    document = load(tmp_path, initial_rollout=1, num_rollouts=3, save_interval=1)
    document["inference"].update(rollout_batch_size=2, samples_per_prompt=2)
    spec = specs.from_dict(document, config_path=FIXTURE)
    root = Path(spec.output["root"])
    debug = root / "debug/train_data"
    debug.mkdir(parents=True)
    # Rollout 0 is the seeding OPD trainer's: teacher log-probs, no verifier evidence. The audit must skip it.
    torch.save(
        {"rollout_data": {"sample_indices": [0], "advantages": [1.0], "teacher_log_probs": [0.0]}}, debug / "0_0.pt"
    )
    log = ""  # this root's training.log starts at the handoff
    for rollout in (1, 2):
        rows = [
            {
                "sample_index": 4 * rollout + i,
                "status": "completed",
                "reward": float(i % 2),
                "metadata": {"r": float(i % 2)},
            }
            for i in range(4)
        ]
        with (root / grpo_hooks.EVIDENCE).open("a") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
        advantages = [torch.tensor([row["reward"] - 0.5]) for row in rows]
        torch.save(
            {"rollout_data": {"sample_indices": [row["sample_index"] for row in rows], "advantages": advantages}},
            debug / f"{rollout}_0.pt",
        )
        log += f"step {rollout}: {{'train/step': {rollout}, 'train/grad_norm': 1.0}}\n"
    (root / "training.log").write_text(log)
    base, export = tmp_path / "base", root / "hf-2"
    base.mkdir()
    export.mkdir()
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([1.0])}, base / "model.safetensors")
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([2.0])}, export / "model.safetensors")
    (export / ".complete").touch()

    async def registered_reward(args, sample, **kwargs):
        return sample.metadata["r"]

    monkeypatch.setattr(grpo_audit.rewards, "registered_reward", registered_reward)
    prepared = {"model": str(base), "data": {"reward_config": "/fixed/registry"}}
    result = grpo_audit.audit(spec, prepared)
    assert [row["rollout_id"] for row in result["rollouts"]] == [1, 2]
    assert [step["train/step"] for step in result["optimizer"]] == [1, 2]
    # A missing step after the handoff still fails.
    (root / "training.log").write_text(log.replace("step 2: {'train/step': 2, 'train/grad_norm': 1.0}\n", ""))
    with pytest.raises(ValueError, match="Missing optimizer step metrics"):
        grpo_audit.audit(spec, prepared)
