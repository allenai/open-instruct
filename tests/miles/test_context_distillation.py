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


def test_isolated_prefill_is_opt_in_and_shared_by_teacher_and_student():
    teacher = pytest.importorskip("open_instruct.miles.distillation.teacher")
    document = recipe()
    ordinary = RunSpec.from_dict(document)
    assert "sglang_prefill_max_requests" not in ordinary.compile().miles
    assert "--prefill-max-requests" not in teacher.command(ordinary.teacher, 123)
    document["teacher"]["isolate_prefill"] = True
    spec = RunSpec.from_dict(document)
    assert spec.compile().miles["sglang_prefill_max_requests"] == 1
    command = teacher.command(spec.teacher, 123)
    assert command[command.index("--prefill-max-requests") + 1] == "1"
    assert command[command.index("--max-running-requests") + 1] == "4"
    assert RunSpec.from_dict(spec.to_dict()).teacher["isolate_prefill"]
    document["miles"] = {"sglang_prefill_max_requests": 2}
    with pytest.raises(ValueError, match="owns miles.sglang_prefill_max_requests"):
        RunSpec.from_dict(document).compile()


@pytest.mark.parametrize("value", [1, "true", None])
def test_isolated_prefill_rejects_non_boolean(value):
    document = recipe()
    document["teacher"]["isolate_prefill"] = value
    with pytest.raises(ValueError, match="isolate_prefill must be boolean"):
        RunSpec.from_dict(document)


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


def test_panel_schedule_skips_intermediate_evaluation(monkeypatch, tmp_path):
    pytest.importorskip("miles")
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"evaluation_rollout_ids": [0, 15, 63]}))
    monkeypatch.setenv("OI_MILES_OPD_PANEL", str(path))
    output = hooks.evaluate(SimpleNamespace(), 31, None, evaluation=True)
    assert output.data == {} and output.metrics["qualification_skipped"] == 1


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


@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize("version", ["0", "16"])
@pytest.mark.parametrize("decode", [[-2.0, -2.0], [-1.5, -3.5]])
def test_two_context_cancellation_is_transported_once_and_keeps_truncated_tokens(
    monkeypatch, calibrate, version, decode
):
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
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", "test")
    monkeypatch.setattr(hooks, "resources", lambda *args: (service, Tokenizer(), asyncio.Semaphore(1)))
    requests = []

    async def post(url, payload, timeout_secs):
        ids = payload["input_ids"]
        requests.append((url, ids))
        # Frozen bare teacher deliberately differs after the student updates.
        scores = (
            [-2.0, -2.0]
            if url == "http://student:123/generate"
            else {11: [-0.5, -3.0], 13: [-0.25, -4.0], 2: [-5.0, -5.0]}[ids[1]]
        )
        entries = [[None if i == 0 else -1.0, t] for i, t in enumerate(ids[:-2])]
        return {
            "meta_info": {
                "input_token_logprobs": entries + [[p, t] for p, t in zip(scores, ids[-2:])],
                "weight_version": version if url == "http://student:123/generate" else "0",
            }
        }

    async def grade(*args, **kwargs):
        return 0.0

    monkeypatch.setattr(hooks.on_policy_distillation, "_post_json", post)
    monkeypatch.setattr(hooks.rewards, "registered_reward", grade)
    sample = SimpleNamespace(
        index=1,
        metadata={"opd_teacher_prompt": "full", "opd_answer_prompt": "answer"},
        tokens=[1, 2, 20, 21],
        response_length=2,
        rollout_log_probs=list(decode),
        train_metadata={},
        weight_versions=[SimpleNamespace(to_dicts=lambda: [{"version": version}])],
        teacher_log_probs=None,
        opd_reverse_kl=None,
        status="truncated",
        loss_mask=[1, 1],
        get_reward_value=lambda args: 0.0,
    )
    args = SimpleNamespace(
        hf_checkpoint="/student", save=None, opd_kl_coef=0.5, sglang_router_ip="student", sglang_router_port=123
    )
    asyncio.run(hooks.reward(args, sample))
    assert len(requests) == (4 if calibrate and version == "0" else 3)
    assert ("http://student:123/generate", [1, 2, 20, 21]) in requests
    assert sample.rollout_log_probs == decode
    assert sample.train_metadata["opd_student_prefill_version"] == version
    assert sample.train_metadata["opd_student_prefill_log_probs"] == [-2.0, -2.0]
    if calibrate and version == "0":
        assert sample.train_metadata["opd_initial_matched_prefill_gap_abs_mean"] == 3.0
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


@pytest.mark.parametrize("scored_version", [None, "0", "17"])
def test_cancellation_rejects_missing_or_stale_student_scoring(monkeypatch, scored_version):
    hooks = pytest.importorskip("open_instruct.miles.distillation.hooks")
    service = dict(
        objective="cancel_answer_positive", endpoint="http://teacher", max_context_length=100, request_timeout=10
    )
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", "test")
    monkeypatch.setattr(
        hooks, "resources", lambda *args: (service, SimpleNamespace(encode=lambda *a, **kw: [1]), asyncio.Semaphore(1))
    )

    async def post(url, payload, **kwargs):
        return {"meta_info": {"weight_version": scored_version, "input_token_logprobs": [[None, 1], [-1.0, 2]]}}

    monkeypatch.setattr(hooks.on_policy_distillation, "_post_json", post)
    sample = SimpleNamespace(
        response_length=1,
        tokens=[1, 2],
        metadata={"opd_teacher_prompt": "full", "opd_answer_prompt": "answer"},
        weight_versions=[SimpleNamespace(to_dicts=lambda: [{"version": "16"}])],
    )
    args = SimpleNamespace(hf_checkpoint="/student", sglang_router_ip="student", sglang_router_port=123)
    with pytest.raises(ValueError, match="does not match rollout"):
        asyncio.run(hooks.reward(args, sample))


def test_identical_prefill_contexts_cancel_without_decode_artifact():
    cancellation = pytest.importorskip("open_instruct.miles.distillation.cancellation")
    bare = [-1.0, -2.0, -3.0]
    decode = [-0.9, -2.1, -2.8]
    pressure, _ = cancellation.token_pressure(bare, bare, bare)
    contaminated, _ = cancellation.token_pressure(decode, bare, bare)
    assert pressure.tolist() == [0.0, 0.0, 0.0]
    assert contaminated.sum().item() < 0
