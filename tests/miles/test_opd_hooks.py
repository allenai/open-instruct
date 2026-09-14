"""Native compatibility checks for rendered text and evaluation reward state."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from miles.utils.data import Dataset

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

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", wrong_scores)
    monkeypatch.setattr(opd_hooks, "_LIMIT", None)
    with pytest.raises(ValueError, match="positions or IDs"):
        asyncio.run(opd_hooks.reward(SimpleNamespace(), SimpleNamespace(tokens=[1, 2, 3], response_length=2)))
