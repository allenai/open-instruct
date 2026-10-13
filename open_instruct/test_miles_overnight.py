"""Contracts for checkpoint-specific screening and matched text diagnostics."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import overnight, panel, qualification_report
from open_instruct.miles.execution import launch


@pytest.mark.parametrize(
    "success,finished,size,expected",
    [
        (False, 6, 8, "finished_wrong"),
        (False, 5, 8, "truncation_heavy"),
        (True, 8, 8, "has_success"),
        (False, 2, 2, "incomplete"),
        (True, 2, 2, "has_success"),
    ],
)
def test_screen_classification(success, finished, size, expected):
    attempts = [
        {"reward": int(success and i == 0), "finish": "stop" if i < finished else "length"} for i in range(size)
    ]
    assert overnight.classify(attempts) == expected


def test_split_requires_minimum_and_is_deterministic_disjoint():
    values = [{"sample_id": str(i)} for i in range(128)]
    a, b = overnight.split(values)
    assert len(a) == 96 and len(b) == 32
    assert (a, b) == overnight.split(list(reversed(values)))
    assert not {x["sample_id"] for x in a} & {x["sample_id"] for x in b}
    a, b = overnight.split(values[:48])
    assert len(a) == 32 and len(b) == 16
    with pytest.raises(ValueError, match="need 48"):
        overnight.split(values[:47])


def test_fixed_text_metric_rewards_the_intended_direction():
    result = panel.movement([-2, -2, -2], [-1, -3, -1], [1, -1, 0])
    assert result["pressure_times_logprob_change_sum"] == 2
    assert result["encouraged_logprob_change_sum"] == 1
    assert result["discouraged_logprob_change_sum"] == -1
    with pytest.raises(ValueError):
        panel.movement([-2], [-1, -3], [1])


def test_pass_at_k_uses_all_draws_without_dropping_failures():
    questions = {"q": [{"reward": int(i == 0)} for i in range(16)]}
    assert qualification_report.pass_at_k(questions, 8) == 0.5
    assert qualification_report.pass_at_k(questions, 1) == 1 / 16
    assert qualification_report.pass_at_k(questions, 16) == 1.0
    assert qualification_report.pass_at_k(questions, 17) is None


def test_overnight_launch_routes_through_preparation():
    document = {
        "schema_version": 1,
        "name": "overnight-test",
        "model": {"source": "/model"},
        "output": {"root": "/output/test"},
        "launch": {"gpus_per_replica": 5, "auto_resume": False, "env": {"OI_MILES_OPD_OVERNIGHT": "1"}},
        "data": {"prompt_data": "/data/train", "reward_config": "/data/rewards"},
        "training": {"filter_zero_std_groups": False},
        "trainer": {"gpus": 2},
        "inference": {"placement_mode": "disaggregated", "gpus": 2, "samples_per_prompt": 2},
        "teacher": {"source": "/model", "objective": "cancel_answer_positive"},
    }
    spec = RunSpec.from_dict(document)
    result = launch.specification("image", spec)
    assert "python -m open_instruct.miles.distillation.overnight" in result["tasks"][0]["arguments"][0]


def test_screen_short_circuits_success_but_requires_eight_for_zero(monkeypatch, tmp_path):
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            return str(tokens[0])

    def post(url, payload, timeout):
        kind = payload["input_ids"][0]
        # Every attempt on kind 1 succeeds, every attempt on kind 2 truncates.
        return {
            "meta_info": {
                "finish_reason": {"type": "length" if kind == 2 else "stop"},
                "output_token_logprobs": [[-1.0, kind]],
            }
        }

    async def reward(args, sample):
        return float(sample.tokens[-1] == 1)

    monkeypatch.setattr(panel, "post", post)
    monkeypatch.setattr(overnight.rewards, "registered_reward", reward)
    items = [
        {
            "sample_id": str(k),
            "previously_all_zero": True,
            "rendered": {"input": "question", "metadata": {"prompt_token_ids": [k]}},
        }
        for k in range(3)
    ]
    eligible, heavy = asyncio.run(
        overnight.screen(items, ["http://fake"], Tokenizer(), tmp_path, time.monotonic() + 30)
    )
    assert [x["sample_id"] for x in eligible] == ["0"]
    assert [x["sample_id"] for x in heavy] == ["2"]
    groups = list(overnight.rows(tmp_path / "screen-groups.jsonl"))
    assert {x["sample_id"]: len(x["screen"]["attempts"]) for x in groups} == {"0": 8, "1": 2, "2": 8}


def test_prepared_cohorts_keep_privileged_context_out_of_eval(tmp_path):
    items = []
    for i in range(48):
        items.append(
            {
                "sample_id": str(i),
                "screen": {"outcome": "finished_wrong"},
                "rendered": {
                    "input": "bare " + str(i),
                    "label": "42",
                    "metadata": {
                        "prepared_sample_id": str(i),
                        "opd_teacher_prompt": "FULL",
                        "opd_answer_prompt": "ANSWER",
                    },
                },
            }
        )
    document = overnight.prepare_run({}, items, [], tmp_path)
    train = list(overnight.rows(tmp_path / "data/train.jsonl"))
    heldout = list(overnight.rows(tmp_path / "data/heldout.jsonl"))
    assert len(train) == 32 and len(heldout) == 16
    assert all("opd_teacher_prompt" in r["metadata"] for r in train)
    assert all("opd_teacher_prompt" not in r["metadata"] for r in heldout)
    assert document["data"]["in_sample_eval"] == ["train_diagnostic"]
    descriptor = json.loads((tmp_path / "panel.json").read_text())
    assert descriptor["evaluation_rollout_ids"] == [0, 15, 63]
    assert len(descriptor["rows"]) == 8


def test_panel_compares_identical_baseline_text_after_policy_change(monkeypatch, tmp_path):
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return {"FULL": [10], "ANSWER": [11]}[text]

    stage = [0]

    def post(url, payload, timeout=600):
        ids = payload["input_ids"]
        value = -1.0 if ids[0] == 10 else -2.0
        if "student" in url and stage[0]:
            value = -1.5
        return {"meta_info": {"input_token_logprobs": [[None if i == 0 else value, t] for i, t in enumerate(ids)]}}

    monkeypatch.setattr(panel, "post", post)
    monkeypatch.setenv("OI_MILES_OPD_TEACHER", json.dumps({"endpoint": "http://teacher"}))
    args = SimpleNamespace(save=str(tmp_path / "checkpoints"), sglang_router_ip="student", sglang_router_port=1)
    samples = [
        SimpleNamespace(
            index=i,
            metadata={"prepared_sample_id": "q"},
            tokens=[1, 2, 3, 4],
            response_length=2,
            rollout_log_probs=[-2.0, -2.0],
            get_reward_value=lambda args: 0.0,
            weight_versions=[],
            status="COMPLETED",
        )
        for i in range(2)
    ]
    output = SimpleNamespace(data={"heldout": {"samples": samples}})
    descriptor = {
        "rows": [
            {"metadata": {"prepared_sample_id": "q", "opd_teacher_prompt": "FULL", "opd_answer_prompt": "ANSWER"}}
        ]
    }
    asyncio.run(panel.observe(args, 0, output, Tokenizer(), descriptor))
    stage[0] = 1
    asyncio.run(panel.observe(args, 15, output, Tokenizer(), descriptor))
    results = list(overnight.rows(tmp_path / "diagnostics/panel-metrics.jsonl"))
    fixed = [r for r in results if r["kind"] == "fixed"]
    assert len(fixed) == 2
    assert all(r["pressure_times_logprob_change_sum"] == 1.0 for r in fixed)
