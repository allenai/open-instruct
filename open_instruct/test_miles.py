"""Contracts that do not require the MILES GPU runtime."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.rewards import rewards


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


@pytest.mark.parametrize("invalid", [{"name": "missing", "target": "42"}, {"name": "known", "weight": "bad"}])
def test_all_reward_specs_are_validated_before_any_scoring(monkeypatch, invalid):
    async def unexpected_score(*args, **kwargs):
        raise AssertionError("Invalid later verifier must be rejected before scoring starts")

    monkeypatch.setattr(rewards, "_registry", lambda path: {"known": SimpleNamespace(async_call=unexpected_score)})
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"))
    sample = SimpleNamespace(metadata={"verifiers": [{"name": "known", "target": "42"}, invalid]})
    with pytest.raises(ValueError):
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
