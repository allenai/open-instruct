"""Contracts for cross-tokenizer OPD through the existing Core learner."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.backends.training_utils.loss_hub import opd

from open_instruct.miles import (
    config,
    core_opd_hooks,
    core_opd_teacher,
    core_opd_training,
    data,
    launch,
    opd_alignment,
    packing,
    specs,
    topology,
)

ROOT = Path(__file__).resolve().parents[2]


def document():
    return specs.load(ROOT / "configs/miles/opd/olmo-moe-tiny.toml").to_dict()


def test_core_dispatch_and_roundtrip():
    spec = specs.from_dict(document())
    assert type(spec).__name__ == "RunSpec"
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    compiled = spec.compile()
    assert compiled.miles["use_opd"]
    assert compiled.miles["opd_log_prob_top_k"] == 0
    assert config.scoring_pass(compiled.core, compiled.miles).standalone
    layout = topology.plan(spec)
    assert layout["policy_gpus"] == 3
    assert layout["teacher_gpus"] == 1
    ray, services = topology.devices(layout["nodes"][0], ["0", "1", "2", "3"])
    assert ray == ["0", "1", "2"] and services["__opd_teacher__"] == ["3"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("teacher", "revision", "main"),
        ("teacher", "gpus", 2),
        ("distillation", "alignment", "auto"),
        ("distillation", "kl_coef", float("nan")),
        ("miles", "use_rollout_logprobs", True),
        ("async", "fully_async", True),
        ("core", "publication_mode", "refresh"),
    ],
)
def test_reject_unsupported_contract(section, key, value):
    doc = document()
    doc.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        specs.from_dict(doc)


def test_forced_score_even_with_one_update_and_zero_reference_kl():
    settings = dict(global_batch_size=8, rollout_batch_size=4, n_samples_per_prompt=2, kl_coef=0)
    assert not config.scoring_pass(config.CoreConfig(), settings).standalone
    assert config.scoring_pass(config.CoreConfig(), settings | {"use_opd": True}).standalone


class Tokenizer:
    all_special_ids = [99]

    def __init__(self, vocab, encoded):
        self.vocab, self.encoded = vocab, encoded

    def decode(self, ids, **kwargs):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode(self, text, **kwargs):
        assert self.decode(self.encoded) == text
        return self.encoded


def test_exact_spans_do_not_align_overlaps_or_utf8_fragments():
    student = Tokenizer({0: b"a", 1: b" b", 2: b"\xc3", 3: b"\xa9", 99: b"<eos>"}, [])
    teacher = Tokenizer({0: b"ctx", 1: b"a", 2: b" ", 3: b"b", 4: b"\xc3\xa9", 99: b"<eos>"}, [0, 1, 2, 3, 4, 99])
    ids, mapping = opd_alignment.align(student, [0, 1, 2, 3, 99], teacher, "ctx")
    assert mapping == [1, None, None, None, None]
    result = {
        "meta_info": {"input_token_logprobs": [[None if i == 0 else -0.5, token] for i, token in enumerate(ids)]}
    }
    scores, mask = opd_alignment.extract_scores(result, ids, mapping)
    assert scores == [-0.5, 0, 0, 0, 0] and mask == [1, 0, 0, 0, 0]
    result["meta_info"]["input_token_logprobs"][1][1] = 100
    with pytest.raises(ValueError, match="IDs"):
        opd_alignment.extract_scores(result, ids, mapping)


def test_prompt_boundary_merge_and_reasoning_termination():
    student = Tokenizer({0: b"x", 1: b"</think>", 2: b"\n"}, [])
    teacher = Tokenizer({0: b"ctxx", 1: b"</think>", 2: b"\n"}, [0, 1, 2])
    assert opd_alignment.align(student, [0, 1, 2], teacher, "ctx")[1] == [None, None, 2]


def rollout(teacher=(-1.0, -9.0)):
    return dict(
        log_probs=[torch.tensor([-2.0, -3.0])],
        teacher_log_probs=[torch.tensor(teacher)],
        metadata=[{"opd_alignment_mask": [1, 0]}],
        loss_masks=[torch.ones(2)],
        response_lengths=[2],
    )


def test_teacher_changes_policy_gradient_without_changing_training_mask():
    gradients = []
    for teacher in ((-1.0, -9.0), (-4.0, -9.0)):
        data = rollout(teacher)
        core_opd_training.prepare(data)
        advantages = [torch.zeros(2)]
        opd.apply_opd_kl_to_advantages(SimpleNamespace(opd_kl_coef=1), data, advantages, data["log_probs"])
        current = torch.zeros(2, requires_grad=True)
        (-current * advantages[0]).sum().backward()
        gradients.append(current.grad)
        assert data["loss_masks"][0].tolist() == [1, 1]
        assert current.grad[1] == 0
    assert gradients[0][0] != gradients[1][0]


@pytest.mark.parametrize("mutation", ["nonfinite", "length", "mask", "missing"])
def test_reject_bad_teacher_before_advantages(mutation):
    data = rollout()
    if mutation == "nonfinite":
        data["teacher_log_probs"][0][0] = float("nan")
    elif mutation == "length":
        data["teacher_log_probs"][0] = torch.ones(3)
    elif mutation == "mask":
        data["metadata"][0]["opd_alignment_mask"] = [2, 0]
    else:
        del data["log_probs"]
    with pytest.raises(ValueError):
        core_opd_training.prepare(data)


@pytest.mark.parametrize("fails", [False, True])
def test_task_eval_restores_teacher_callback(monkeypatch, fails):
    args = SimpleNamespace(custom_rm_path="teacher", custom_reward_post_process_path="post")

    def generate(received, *args, **kwargs):
        assert received.custom_rm_path.endswith("registered_reward")
        assert received.custom_reward_post_process_path is None
        if fails:
            raise RuntimeError("eval failed")
        return 42

    monkeypatch.setattr(core_opd_hooks, "generate_rollout", generate)
    if fails:
        with pytest.raises(RuntimeError):
            core_opd_hooks.evaluate(args, 0, None, evaluation=True)
    else:
        assert core_opd_hooks.evaluate(args, 0, None, evaluation=True) == 42
    assert args.custom_rm_path == "teacher" and args.custom_reward_post_process_path == "post"


def test_teacher_command_keeps_tp_and_source_independent():
    service = document()["teacher"] | {"snapshot": "/models/arbitrary-teacher"}
    cmd = core_opd_teacher.command(service, 12345)
    assert cmd[cmd.index("--model-path") + 1] == "/models/arbitrary-teacher"
    assert cmd[cmd.index("--tp") + 1] == "1"


def test_packing_preserves_teacher_alignment_and_samples():
    batch = rollout()
    batch["tokens"] = [torch.tensor([7, 8, 9])]
    batch["total_lengths"] = [3]
    core_opd_training.prepare(batch)
    samples = data.sample_batches(batch, 10)
    packed = packing.combine(samples + samples, [0, 1])
    assert packed["tokens"].tolist() == [[7, 8, 9, 7, 8, 9]]
    assert [m["opd_alignment_mask"] for m in packed["metadata"]] == [[1, 0], [1, 0]]
    assert [v.tolist() for v in packed["opd_reverse_kl"]] == [[-1.0, 0.0], [-1.0, 0.0]]


def test_teacher_launch_uses_supervisor_and_host_network():
    spec = specs.from_dict(document())
    task = launch.specification("test-image", spec)["tasks"][0]
    assert task["hostNetworking"]
    assert task["resources"]["gpuCount"] == 4
    assert "open_instruct.miles.cluster" in task["arguments"][0]
