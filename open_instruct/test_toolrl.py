"""Offline call grading, split isolation, and MILES adoption without model loading."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.data.rlvr import toolrl

from open_instruct import toolrl_utils
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import rewards


def completion(calls=None, *, reasoning="plan", response="answer"):
    body = (
        "<response>" + response + "</response>"
        if calls is None
        else "<tool_call>\n" + "\n".join(json.dumps(call) for call in calls) + "\n</tool_call>"
    )
    return f"<think>{reasoning}</think>\n{body}"


def call(platform):
    return {"name": "giveaways", "parameters": {"platform": platform}}


def test_parallel_call_rewards_and_order():
    expected = completion([call("xbox"), call("android")])
    correct = toolrl_utils.score_response(completion([call("android"), call("xbox")]), expected)
    missing = toolrl_utils.score_response(completion([call("xbox")]), expected)
    duplicated = toolrl_utils.score_response(completion([call("xbox"), call("xbox")]), expected)
    wrong = toolrl_utils.score_response(completion([{"name": "chess", "parameters": {}}]), expected)
    assert correct.score == 4
    assert wrong.score < missing.score < correct.score
    assert duplicated.correctness_score < correct.correctness_score
    assert missing.correctness_score == pytest.approx(0)


def test_no_call_targets_only_check_format_not_answer_facts():
    expected = completion(response="a reference answer")
    result = toolrl_utils.score_response(completion(response="a different answer"), expected)
    assert result.score == result.format_score == 1
    assert result.correctness_score == 0
    assert toolrl_utils.score_response(completion([call("xbox")]), expected).score == 0


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"name":"giveaways","parameters":{"platform":"xbox","platform":"android"}}',
        '{"name":"giveaways","parameters":{"platform":NaN}}',
        '{"name":"giveaways","parameters":[]}',
        '{"name":"giveaways","parameters":{},"extra":true}',
        "",
    ],
)
def test_malformed_calls_cannot_earn_correctness(payload):
    prediction = "<think>plan</think>\n<tool_call>\n" + payload + "\n</tool_call>"
    result = toolrl_utils.score_response(prediction, completion([call("xbox")]))
    assert result.correctness_score == -3


def test_reasoning_only_and_multiple_blocks_do_not_supply_actions():
    reference = completion([call("xbox")])
    quoted = reference.split("</think>\n")[1]
    for prediction in (completion(reasoning=quoted), "<think>" + quoted, reference + "\n" + quoted):
        assert toolrl_utils.score_response(prediction, reference).correctness_score == -3


def test_forced_think_prefix_is_explicit():
    reference = completion([call("xbox")])
    suffix = reference.removeprefix("<think>")
    assert toolrl_utils.score_response(suffix, reference).format_score == 0
    assert toolrl_utils.score_response(suffix, reference, think_prefix_in_prompt=True).score == 4
    assert toolrl_utils.score_response(reference, reference, think_prefix_in_prompt=True).format_score == 0


def test_bad_target_raises_instead_of_punishing_model():
    with pytest.raises(ValueError, match="reference"):
        toolrl_utils.score_response("anything", "malformed reference")


def source_rows():
    rows = [
        {"instruction": "tool menu", "input": f"question {index}", "output": completion([call(str(index))])}
        for index in range(12)
    ]
    duplicate = copy.deepcopy(rows[0])
    duplicate["instruction"] = "a different menu"
    rows.append(duplicate)
    return rows


def test_preparation_preserves_source_and_keeps_targets_out_of_prompt():
    source = source_rows()
    before = copy.deepcopy(source)
    partitions, summary = toolrl.convert_rows(source, eval_count=3)
    assert source == before
    assert sum(map(len, partitions.values())) == len(source)
    train = {row["messages"][1]["content"] for row in partitions["train"]}
    evaluation = {row["messages"][1]["content"] for row in partitions["eval"]}
    assert not train & evaluation
    assert summary["unique_dialogues"] == 12
    assert toolrl.convert_rows(source, eval_count=3) == (partitions, summary)
    for rows in partitions.values():
        for row in rows:
            original = source[row["metadata"]["source_row"]]
            assert row["messages"] == [
                {"role": "system", "content": original["instruction"]},
                {"role": "user", "content": original["input"]},
            ]
            assert row["ground_truth"] == row["metadata"]["verifiers"][0]["target"] == original["output"]


def test_source_digest_checked_before_writing(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        toolrl.prepare(b"[]", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_invalid_reference_is_recorded_without_renumbering_source_rows():
    source = source_rows()
    source[2]["output"] = completion([{"name": "", "parameters": {}}])
    partitions, summary = toolrl.convert_rows(source, eval_count=3)
    indices = {row["metadata"]["source_row"] for rows in partitions.values() for row in rows}
    assert indices == set(range(len(source))) - {2}
    assert summary["retained_rows"] == len(source) - 1
    assert summary["rejected_rows"][0]["source_row"] == 2


class Tokenizer:
    def __init__(self, prefix):
        self.chat_template = "template"
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages) + "|assistant|" + self.prefix

    def encode(self, text, **kwargs):
        return list(text.encode())


@pytest.mark.parametrize("prefix", ["", "<think>"])
def test_manifest_to_miles_to_reward_and_resume(tmp_path, monkeypatch, prefix):
    # Use the production writer on an explicitly substituted, local test source.
    source = source_rows() * 308
    source = source[:4000]
    raw = json.dumps(source).encode()
    monkeypatch.setattr(toolrl, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    root = tmp_path / "source"
    toolrl.prepare(raw, root, eval_count=500, think_prefix_in_prompt=bool(prefix))
    hf = tmp_path / "hf"
    hf.mkdir()
    tokenizer = Tokenizer(prefix)
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: tokenizer)
    data = {"rl_manifest": str(root / "rl-manifest.json")}
    result = run_data.prepare_data(data, hf, tmp_path / "prepared", max_prompt_length=5000, seed=17)
    assert run_data.prepare_data(data, hf, tmp_path / "prepared", max_prompt_length=5000, seed=17) == result
    row = json.loads(Path(result["prompt_data"]).read_text().splitlines()[0])
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            reward_config=result["reward_config"], reward_zero_truncated=True, reward_final_answer_only=False
        )
    )
    sample = SimpleNamespace(
        prompt=row["input"],
        response=row["label"].removeprefix(prefix) if prefix else row["label"],
        metadata=row["metadata"],
        tokens=[],
        response_length=0,
        status="completed",
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 4
    tokenizer.prefix = "" if prefix else "<think>"
    with pytest.raises(InputError, match="generation prefix"):
        run_data.prepare_data(data, hf, tmp_path / "wrong-prefix", max_prompt_length=5000, seed=17)
    tokenizer.prefix = "<think>\n"
    with pytest.raises(InputError, match="whitespace"):
        run_data.prepare_data(data, hf, tmp_path / "whitespace-prefix", max_prompt_length=5000, seed=17)
    with pytest.raises(FileExistsError):
        toolrl.prepare(raw, root, eval_count=500)
