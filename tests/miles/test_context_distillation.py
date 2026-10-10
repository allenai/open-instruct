"""Check the context boundary, objective sign, and GPU ownership for Core OPD."""

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import topology
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import alignment


def recipe():
    return {
        "schema_version": 1,
        "name": "context-distillation-test",
        "model": {"source": "/models/student"},
        "output": {"root": "/output/test"},
        "launch": {"gpus_per_replica": 5, "auto_resume": False},
        "data": {"prompt_data": "/data/train.jsonl", "reward_config": "/data/verifiers.json"},
        "training": {"filter_zero_std_groups": False},
        "trainer": {"gpus": 2},
        "inference": {"placement_mode": "disaggregated", "gpus": 2, "samples_per_prompt": 2},
        "teacher": {"source": "/models/teacher"},
    }


def test_context_tokens_are_excluded_and_response_ids_are_exact():
    ids = [10, 11, 12, 20, 21]
    result = {"meta_info": {"input_token_logprobs": [[None, 10], [-1, 11], [-2, 12], [-3, 20], [-4, 21]]}}
    assert alignment.extract_scores(result, ids, 2) == [-3, -4]
    with pytest.raises(ValueError, match="different token"):
        alignment.extract_scores(result, [10, 11, 12, 20, 22], 2)
    result["meta_info"]["input_token_logprobs"][-1][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        alignment.extract_scores(result, ids, 2)


def test_teacher_allocation_and_configuration_round_trip():
    spec = RunSpec.from_dict(recipe(), config_path=Path("/tmp/run.json"))
    config = spec.compile()
    assert config.miles["use_opd"]
    assert not config.miles["normalize_advantages"]
    assert config.core.scoring_pass_required
    assert not config.core.filter_zero_std_groups
    assert RunSpec.from_dict(spec.to_dict()).teacher == spec.teacher
    layout = topology.plan(spec)
    assert layout["allocated_gpus"] == 5
    assert layout["policy_gpus"] == 4
    assert layout["teacher_gpus"] == 1
    assert topology.devices(layout["nodes"][0], [str(i) for i in range(5)]) == (
        ["0", "1", "2", "3"],
        {"__opd_teacher__": ["4"]},
    )


@pytest.mark.parametrize(
    "change",
    [
        {"training": {"filter_zero_std_groups": True}},
        {"miles": {"rollout_temperature": 0.8}},
        {"miles": {"normalize_advantages": True}},
        {"async": {"fully_async": True}},
    ],
)
def test_rejects_incompatible_objective_or_collection(change):
    document = recipe()
    document.update(change)
    with pytest.raises(ValueError):
        RunSpec.from_dict(document)


def test_native_opd_advantage_and_gradient_direction():
    torch = pytest.importorskip("torch")
    opd = pytest.importorskip("miles.backends.training_utils.loss_hub.opd")
    old = torch.tensor([math.log(0.25), math.log(0.75)])
    teacher = torch.tensor([math.log(0.5), math.log(0.5)], requires_grad=True)
    advantages = [torch.zeros(2)]
    args = SimpleNamespace(opd_kl_coef=1.0)
    rollout = {"teacher_log_probs": [teacher]}
    opd.apply_opd_kl_to_advantages(args, rollout, advantages, [old])
    assert torch.allclose(advantages[0], teacher.detach() - old)
    assert not advantages[0].requires_grad
    current = old.clone().requires_grad_(True)
    loss = -(torch.exp(current - old) * advantages[0]).mean()
    loss.backward()
    assert current.grad[0] < 0  # Gradient descent raises the teacher-preferred token.
    assert current.grad[1] > 0
    assert teacher.grad is None


def test_privileged_context_is_scoring_only_and_truncated_loss_is_retained(monkeypatch):
    pytest.importorskip("miles")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "question plus successful solution"
            assert not add_special_tokens
            return [10, 11, 12]

    service = {"max_context_length": 100, "request_timeout": 10, "source": "/frozen", "endpoint": "http://teacher"}
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", "test")
    monkeypatch.setattr(hooks, "resources", lambda *args: (service, Tokenizer(), asyncio.Semaphore(1)))

    async def post(url, payload, timeout_secs):
        assert payload["input_ids"] == [10, 11, 12, 20, 21]
        assert payload["sampling_params"]["max_new_tokens"] == 0
        return {"meta_info": {"input_token_logprobs": [[None, 10], [-1, 11], [-2, 12], [-3, 20], [-4, 21]]}}

    async def grade(*args, **kwargs):
        return 0.0

    monkeypatch.setattr(hooks.on_policy_distillation, "_post_json", post)
    monkeypatch.setattr(hooks.rewards, "registered_reward", grade)
    sample = SimpleNamespace(
        index=1,
        metadata={"opd_teacher_prompt": "question plus successful solution"},
        tokens=[1, 2, 20, 21],
        response_length=2,
        train_metadata={},
        weight_versions=[],
        teacher_log_probs=None,
        status="truncated",
        loss_mask=[1, 1],
        get_reward_value=lambda args: 0.0,
    )
    args = SimpleNamespace(hf_checkpoint="/student", save=None)
    assert asyncio.run(hooks.reward(args, sample)) == 0.0
    assert sample.tokens == [1, 2, 20, 21]
    assert sample.teacher_log_probs.tolist() == [-3.0, -4.0]
    assert sample.loss_mask == [1, 1]
    assert hooks.post_process(args, [sample]) == ([0.0], [0.0])
