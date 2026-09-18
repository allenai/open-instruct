"""Native compatibility checks for rendered text and evaluation reward state."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from miles.utils.data import Dataset
from miles.utils.types import Sample

from open_instruct.miles import opd_hooks


def test_native_dataset_preserves_rendered_text_with_processor(tmp_path):
    prompt = "<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"input": prompt, "label": "2"}) + "\n")
    dataset = Dataset(
        str(path),
        tokenizer=None,
        processor=object(),
        max_length=None,
        prompt_key="input",
        label_key="label",
        apply_chat_template=False,
    )
    assert dataset[0].prompt == prompt
    assert dataset[0].multimodal_inputs is None


@pytest.mark.parametrize("fails", [False, True])
def test_evaluation_restores_cached_training_reward(monkeypatch, fails):
    args = SimpleNamespace(custom_rm_path="teacher", custom_reward_post_process_path="teacher-post")
    captured = []

    def generate(received, *positional, **kwargs):
        captured.append(received)
        assert received.custom_rm_path.endswith("eval_reward")
        if fails:
            raise RuntimeError("evaluation failed")
        return "result"

    monkeypatch.setattr(opd_hooks.sglang_rollout, "generate_rollout", generate)
    if fails:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            opd_hooks.evaluate(args, 0, None, evaluation=True)
    else:
        assert opd_hooks.evaluate(args, 0, None, evaluation=True) == "result"
    assert captured[0] is args
    assert captured[0].custom_rm_path == "teacher"
    assert args.custom_reward_post_process_path == "teacher-post"


def test_teacher_token_ids_must_match(monkeypatch):
    async def wrong_scores(*args, **kwargs):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 99]]}}

    monkeypatch.delenv("OI_OPD_EOPD_TOP_K", raising=False)
    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", wrong_scores)
    monkeypatch.setattr(opd_hooks, "_LIMIT", None)
    with pytest.raises(ValueError, match="positions or IDs"):
        asyncio.run(
            opd_hooks.reward(SimpleNamespace(), SimpleNamespace(tokens=[1, 2, 3], response_length=2, metadata={}))
        )


def test_reward_is_numeric_and_post_process_extracts_the_teacher_scores(monkeypatch, tmp_path):
    """Miles's rollout metrics round and group by sample.reward, so the payload lives in metadata."""

    async def scores(*args, **kwargs):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]]}}

    monkeypatch.delenv("OI_OPD_EOPD_TOP_K", raising=False)
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    monkeypatch.setattr(opd_hooks, "_LIMIT", None)
    sample = Sample(index=1, tokens=[1, 2, 3], response_length=2, response="ab")
    sample.reward = asyncio.run(opd_hooks.reward(SimpleNamespace(), sample))
    assert sample.reward == 0.0 and round(sample.reward, 1) == 0.0
    assert opd_hooks.TEACHER_RESPONSE_KEY in sample.metadata
    rewards, raw = opd_hooks.post_process(SimpleNamespace(reward_key=None), [sample])
    assert rewards == raw == [0.0]
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-1.0, -2.0]))
    assert opd_hooks.TEACHER_RESPONSE_KEY not in sample.metadata
    record = json.loads((tmp_path / "teacher-scores.jsonl").read_text().splitlines()[0])
    assert record["teacher_log_probs"] == [-1.0, -2.0] and "eopd_gate" not in record
    with pytest.raises(ValueError, match="no teacher response"):
        opd_hooks.post_process(SimpleNamespace(reward_key=None), [Sample(tokens=[1, 2], response_length=1)])


TEACHER_TOP_K = {
    "meta_info": {
        "input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]],
        "input_top_logprobs": [None, [[-0.1, 2], [-2.0, 7]], [[-0.5, 3], [-0.9, 8]]],
    }
}


def _eopd_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OI_OPD_EOPD_TOP_K", "2")
    monkeypatch.setenv("OI_OPD_EOPD_TAU", "0.5")
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    monkeypatch.setattr(opd_hooks, "_LIMIT", None)


def test_eopd_reward_requests_and_checks_the_teacher_top_k(monkeypatch, tmp_path):
    _eopd_environment(monkeypatch, tmp_path)
    payloads = []

    async def post(url, payload, timeout_secs=None):
        payloads.append((url, payload, timeout_secs))
        return TEACHER_TOP_K

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "_post_json", post)
    args = SimpleNamespace(rm_url="http://teacher/generate", sglang_router_request_timeout_secs=7)
    sample = SimpleNamespace(tokens=[1, 2, 3], response_length=2, metadata={})
    assert asyncio.run(opd_hooks.reward(args, sample)) == 0.0
    assert sample.metadata[opd_hooks.TEACHER_RESPONSE_KEY] is TEACHER_TOP_K
    assert payloads[0][0] == "http://teacher/generate" and payloads[0][2] == 7
    assert payloads[0][1]["top_logprobs_num"] == 2 and payloads[0][1]["input_ids"] == [1, 2, 3]

    async def short(url, payload, timeout_secs=None):
        return {"meta_info": {**TEACHER_TOP_K["meta_info"], "input_top_logprobs": [None, [[-0.1, 2]], [[-0.5, 3]]]}}

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "_post_json", short)
    with pytest.raises(ValueError, match="expected 2"):
        asyncio.run(opd_hooks.reward(args, sample))


def test_eopd_post_process_stores_the_top_k_for_training(monkeypatch, tmp_path):
    _eopd_environment(monkeypatch, tmp_path)
    sample = Sample(index=3, tokens=[1, 2, 3], response_length=2, response="ab", reward=0.0)
    sample.metadata[opd_hooks.TEACHER_RESPONSE_KEY] = TEACHER_TOP_K
    args = SimpleNamespace(reward_key=None, opd_log_prob_top_k=0)
    assert opd_hooks.post_process(args, [sample]) == ([0.0], [0.0])
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-1.0, -2.0]))
    assert sample.train_metadata["eopd_topk_ids"] == [[2, 7], [3, 8]]
    assert sample.train_metadata["eopd_topk_logprobs"] == [[-0.1, -2.0], [-0.5, -0.9]]
    record = json.loads((tmp_path / "teacher-scores.jsonl").read_text().splitlines()[0])
    # Position 0 is nearly one-hot (gate off at tau 0.5); position 1 is close to even (gate on).
    assert record["eopd_gate"] == [0, 1]
    assert len(record["eopd_proxy_entropy"]) == len(record["eopd_topk_mass"]) == 2
    assert record["teacher_log_probs"] == [-1.0, -2.0]
