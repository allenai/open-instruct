"""Contracts that do not require the MILES GPU runtime."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from open_instruct.miles import rewards
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.data import policy_versions, sample_batches, score_agreement, validate_score_agreement
from open_instruct.miles.state import PolicyClock, atomic_json


def test_config_compiles_native_miles_options(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        '[miles]\nhf_checkpoint="model"\nglobal_batch_size=8\nrollout_batch_size=2\nn_samples_per_prompt=4\n'
    )
    args = RunConfig.load(path).arguments()
    assert args[args.index("--train-backend") + 1] == "olmo_core"
    assert "--no-offload-train" in args
    assert args[args.index("--global-batch-size") + 1] == "8"
    assert "--olmo-core-config" in args


@pytest.mark.parametrize(
    "change",
    [
        {"train_backend": "fsdp"},
        {"global_batch_size": 0},
        {"micro_batch_size": 2},
        {"fully_async": True},
        {"offload_train": True},
        {"qkv_format": "thd"},
        {"use_rollout_routing_replay": True},
        {"check_weight_update_selector": "target"},
    ],
)
def test_config_rejects_unimplemented_or_ambiguous_behavior(change):
    with pytest.raises(ValueError):
        RunConfig(CoreConfig(), {"hf_checkpoint": "model", "global_batch_size": 8, **change}).arguments()


@pytest.mark.parametrize("prompt_limit", [1536, 2048])
def test_prompt_limit_leaves_room_for_a_response(prompt_limit):
    options = dict(hf_checkpoint="model", global_batch_size=8, rollout_max_context_len=1536)
    RunConfig(CoreConfig(), {**options, "rollout_max_prompt_len": 1024}).validate()
    with pytest.raises(ValueError, match="must be smaller"):
        RunConfig(CoreConfig(), {**options, "rollout_max_prompt_len": prompt_limit}).validate()


def test_clock_counts_optimizer_steps_and_republication():
    clock = PolicyClock()
    clock.published()
    assert clock.validate_versions([0], 0) == 0
    clock.optimizer_step(True)
    clock.optimizer_step(True)
    clock.published()
    assert clock.published_step == 2
    clock.published()
    assert clock.published_step == 2
    assert PolicyClock.from_dict(clock.as_dict()) == clock
    with pytest.raises(ValueError, match="lag"):
        clock.validate_versions([0], 1)
    with pytest.raises(RuntimeError, match="skipped"):
        clock.optimizer_step(False)
    assert clock.completed_steps == 2


@pytest.mark.parametrize("versions", [[], [-1], [True], [3]])
def test_clock_refuses_missing_invalid_future_versions(versions):
    with pytest.raises(ValueError):
        PolicyClock(completed_steps=2, published_step=2).validate_versions(versions, 10)


def test_tool_tokens_keep_masks_and_behavior_anchor():
    original = {
        "tokens": [torch.tensor([1, 2, 3, 4, 5])],
        "total_lengths": [5],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1, 0, 1])],
        "weight_versions": [["0", "0"]],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2, -0.3])],
    }
    batch = sample_batches(original, 8)[0]
    assert batch["tokens"].shape == (1, 5)
    assert batch["loss_masks"][0].tolist() == [1, 0, 1]
    assert batch["rollout_log_probs"][0] is original["rollout_log_probs"][0]
    assert policy_versions(batch) == [0, 0]
    with pytest.raises(ValueError):
        sample_batches(original, 4)


def test_atomic_manifest(tmp_path):
    path = tmp_path / "run" / "complete.json"
    atomic_json(path, {"step": 2})
    assert json.loads(path.read_text()) == {"step": 2}
    assert list(path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("value", [-1, True, 1.5, "-1", "1.0", "12"])
def test_invalid_policy_metadata_rejected(value):
    payload = value if value == "12" else [value]
    with pytest.raises(ValueError):
        policy_versions({"weight_versions": [payload]})


def test_mixed_rewards_use_existing_verifier_and_preserve_components(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps({"math_answer": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}}))
    sample = SimpleNamespace(
        tokens=[1, 2, 3],
        response_length=1,
        response="The answer is 42",
        prompt="6 times 7?",
        metadata={
            "verifiers": [
                {"name": "math_answer", "target": "42", "weight": 0.75},
                {"name": "math_answer", "target": "43", "weight": 0.25},
            ]
        },
    )
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(path)))
    assert asyncio.run(rewards.registered_reward(args, [sample])) == [0.75]
    assert [item["score"] for item in sample.metadata["reward_components"]] == [1.0, 0.0]
    sample.metadata["verifiers"][0]["name"] = "untrusted.module.Factory"
    with pytest.raises(ValueError, match="trusted reward registry"):
        asyncio.run(rewards.registered_reward(args, sample))


def test_dictionary_targets_survive_repeated_and_shared_ifeval_scoring(tmp_path):
    registry = tmp_path / "verifiers.json"
    registry.write_text(json.dumps({"ifeval": {"factory": "open_instruct.ground_truth_utils.IFEvalVerifierOld"}}))
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(registry)))
    target = {"func_name": "validate_lowercase", "N": None}
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=response,
            prompt="Respond in lowercase.",
            metadata={"verifiers": [{"name": "ifeval", "target": target}]},
        )
        for response in ("the ocean is blue.", "another lowercase answer.", "The ocean is blue.")
    ]

    async def score_twice():
        for _ in range(2):
            assert await rewards.registered_reward(args, samples) == [1.0, 1.0, 0.0]
            assert target == {"func_name": "validate_lowercase", "N": None}
            assert all(sample.metadata["verifiers"][0]["target"] is target for sample in samples)
            assert [sample.metadata["reward_components"][0]["score"] for sample in samples] == [1.0, 1.0, 0.0]

    asyncio.run(score_twice())


def test_policy_agreement_ignores_tool_tokens_and_weights_active_tokens():
    rollout = {
        "log_probs": [torch.tensor([-1.1, float("nan"), -1.2]), torch.tensor([-2.3])],
        "rollout_log_probs": [torch.tensor([-1.0, float("nan"), -1.0]), torch.tensor([-2.0])],
        "loss_masks": [torch.tensor([1, 0, 1]), torch.tensor([1])],
    }
    stats = score_agreement(rollout)
    assert stats.tolist() == pytest.approx([0.6, 3])
    assert validate_score_agreement(stats, 0.21) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="exceeds"):
        validate_score_agreement(stats, 0.19)
    rollout["loss_masks"][0][1] = 1
    with pytest.raises(ValueError, match="Non-finite"):
        score_agreement(rollout)
