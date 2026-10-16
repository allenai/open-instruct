"""Check the context boundary, objective sign, and GPU ownership for Core OPD."""

import asyncio
import json
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


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_is_unhinted_logged_and_restores_training_hooks(monkeypatch, tmp_path, fail):
    pytest.importorskip("miles")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    args = SimpleNamespace(
        custom_rm_path="teacher.reward",
        custom_reward_post_process_path="teacher.post",
        save=str(tmp_path / "checkpoints"),
    )
    sample = SimpleNamespace(
        index=0,
        metadata={"prepared_sample_id": "q1"},
        response="Answer: 42",
        response_length=4,
        status="COMPLETED",
        weight_versions=[],
        get_reward_value=lambda args: 1.0,
    )
    expected = SimpleNamespace(data={"heldout": {"samples": [sample]}})

    def generate(arguments, rollout_id, data_source, evaluation):
        assert arguments.custom_rm_path.endswith("registered_reward")
        assert arguments.custom_reward_post_process_path is None
        assert evaluation
        if fail:
            raise RuntimeError("evaluation failed")
        return expected

    monkeypatch.setattr(hooks.importlib, "import_module", lambda name: SimpleNamespace(generate_rollout=generate))
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            hooks.evaluate(args, 0, None, evaluation=True)
    else:
        assert hooks.evaluate(args, 0, None, evaluation=True) is expected
        record = json.loads((tmp_path / "qualification-evaluation.jsonl").read_text())
        assert record["sample_id"] == "q1" and record["reward"] == 1.0
        assert record["response"] == "Answer: 42"
    assert args.custom_rm_path == "teacher.reward"
    assert args.custom_reward_post_process_path == "teacher.post"


def test_privileged_context_is_scoring_only_and_truncated_loss_is_retained(monkeypatch):
    pytest.importorskip("miles")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "question plus successful solution"
            assert not add_special_tokens
            return [10, 11, 12]

    service = {"max_context_length": 100, "request_timeout": 10, "source": "/frozen", "endpoint": "http://teacher"}
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", json.dumps(service))
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


@pytest.mark.parametrize("calibrate", [False, True])
def test_two_context_cancellation_is_transported_once_and_keeps_truncated_tokens(monkeypatch, calibrate):
    pytest.importorskip("miles")
    torch = pytest.importorskip("torch")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    opd = pytest.importorskip("miles.backends.training_utils.loss_hub.opd")

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return {"full": [10, 11, 12], "answer": [10, 13]}[text]

    service = {
        "max_context_length": 100,
        "request_timeout": 10,
        "source": "/frozen",
        "endpoint": "http://teacher",
        "objective": "cancel_answer_positive",
        "kl_coef": 0.5,
        "calibrate_initial": calibrate,
    }
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", json.dumps(service))
    monkeypatch.setattr(hooks, "resources", lambda *args: (service, Tokenizer(), asyncio.Semaphore(1)))
    requests = []

    async def post(url, payload, timeout_secs):
        ids = payload["input_ids"]
        requests.append(ids)
        scores = {11: [-0.5, -3.0], 13: [-0.25, -4.0], 2: [-2.0, -2.0]}[ids[1]]
        entries = [[None if i == 0 else -1.0, t] for i, t in enumerate(ids[:-2])]
        return {"meta_info": {"input_token_logprobs": entries + [[p, t] for p, t in zip(scores, ids[-2:])]}}

    async def grade(*args, **kwargs):
        return 0.0

    monkeypatch.setattr(hooks.on_policy_distillation, "_post_json", post)
    monkeypatch.setattr(hooks.rewards, "registered_reward", grade)
    sample = SimpleNamespace(
        index=1,
        metadata={"opd_teacher_prompt": "full", "opd_answer_prompt": "answer"},
        tokens=[1, 2, 20, 21],
        response_length=2,
        rollout_log_probs=[-2.0, -2.0],
        train_metadata={},
        weight_versions=[SimpleNamespace(to_dicts=lambda: [{"version": "0"}])],
        teacher_log_probs=None,
        opd_reverse_kl=None,
        status="truncated",
        loss_mask=[1, 1],
        get_reward_value=lambda args: 0.0,
    )
    args = SimpleNamespace(hf_checkpoint="/student", save=None, opd_kl_coef=0.5)
    asyncio.run(hooks.reward(args, sample))
    assert len(requests) == (3 if calibrate else 2)
    if calibrate:
        assert sample.train_metadata["opd_initial_bare_gap_abs_max"] == 0.0
    assert sample.tokens == [1, 2, 20, 21] and sample.loss_mask == [1, 1]
    assert sample.opd_reverse_kl == [0.0, 1.0]
    assert hooks.post_process(args, [sample]) == ([0.0], [0.0])
    advantages = [torch.zeros(2)]
    # A different Core scoring anchor must not recompute or double-add the pressure.
    opd.apply_opd_kl_to_advantages(
        args,
        {"opd_reverse_kl": [torch.tensor(sample.opd_reverse_kl)], "teacher_log_probs": [sample.teacher_log_probs]},
        advantages,
        [torch.tensor([-1.9, -2.1])],
    )
    assert advantages[0].tolist() == [0.0, -0.5]
    assert not advantages[0].requires_grad


@pytest.mark.parametrize(
    "behavior,expected",
    [
        ([-3.0, -1.0, -3.0, -3.0, -0.5, -2.0], [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
        ([-1.0, -5.0, -5.0, -2.0], [0.0, 0.5, 0.0, 0.0]),
        ([-3.0], [0.5]),
        ([-1.0, -2.0, -3.0, -4.0, -5.0], [0.0, 0.0, 0.0, 0.0, 0.5]),
    ],
)
def test_tail_negative_selection_is_per_response_and_stable(behavior, expected):
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    assert hooks.tail_negative(behavior, len(behavior)) == expected


@pytest.mark.parametrize(
    "behavior,length", [(None, 1), ([], 0), ([-1.0], 2), ([float("nan")], 1), ([float("-inf")], 1), ([0.1], 1)]
)
def test_tail_negative_rejects_missing_or_invalid_behavior(behavior, length):
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    with pytest.raises(ValueError):
        hooks.tail_negative(behavior, length)


def test_tail_negative_hook_has_no_teacher_and_native_advantages_are_negative(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    opd = pytest.importorskip("miles.backends.training_utils.loss_hub.opd")
    service = hooks.config.parse({"teacher": {"objective": "tail_negative"}}, tmp_path)
    monkeypatch.setenv(hooks.config.ENV, json.dumps(service))

    def forbidden(*args, **kwargs):
        pytest.fail("Tail-negative must not load a teacher/tokenizer or request teacher scores")

    async def observed_reward(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(hooks, "resources", forbidden)
    monkeypatch.setattr(hooks.on_policy_distillation, "_post_json", forbidden)
    monkeypatch.setattr(hooks.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(hooks.rewards, "registered_reward", observed_reward)
    args = SimpleNamespace(save=str(tmp_path / "checkpoints"), opd_kl_coef=1.0)
    samples = []
    for index, behavior in enumerate([[-3.0, -1.0, -3.0, -3.0, -0.5, -2.0], [-2.0]]):
        length = len(behavior)
        sample = SimpleNamespace(
            index=index,
            metadata={},
            tokens=[7, 8] + list(range(length)),
            response_length=length,
            rollout_log_probs=behavior,
            teacher_log_probs=None,
            opd_reverse_kl=None,
            train_metadata={},
            loss_mask=[1] * length,
            status="TRUNCATED",
            weight_versions=[],
            get_reward_value=lambda args: 1.0,
        )
        assert asyncio.run(hooks.reward(args, sample)) == 1.0
        assert sample.teacher_log_probs is None
        assert sample.tokens == [7, 8] + list(range(length)) and sample.loss_mask == [1] * length
        samples.append(sample)
    assert hooks.post_process(args, samples) == ([1.0, 1.0], [0.0, 0.0])
    transported = [torch.tensor(s.opd_reverse_kl, requires_grad=True) for s in samples]
    advantages = [torch.zeros(s.response_length) for s in samples]
    opd.apply_opd_kl_to_advantages(
        args, {"opd_reverse_kl": transported}, advantages, [torch.full((s.response_length,), -99.0) for s in samples]
    )
    assert advantages[0].tolist() == [-0.5, 0.0, -0.5, 0.0, 0.0, 0.0]
    assert advantages[1].tolist() == [-0.5]
    assert all(not adv.requires_grad for adv in advantages)
    records = [json.loads(line) for line in (tmp_path / "teacher-scores.jsonl").read_text().splitlines()]
    assert all("teacher_log_probs" not in record and "opd_teacher_source" not in record for record in records)
    assert records[0]["opd_reverse_kl"] == samples[0].opd_reverse_kl
    for pressure in (None, [0.5], [float("nan")] * 6, [-0.5] * 6, [0.5] * 6, [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]):
        samples[0].opd_reverse_kl = pressure
        with pytest.raises(ValueError, match="Tail-negative pressure"):
            hooks.post_process(args, samples)
