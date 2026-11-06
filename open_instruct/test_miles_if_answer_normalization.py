"""Exercise real native IF; opt into a separately qualified module in its image.

Set OPEN_INSTRUCT_TEST_QUALIFIED_IF_MODULE to run the same parity cases against
that module. Source qualification failures must propagate, never be patched out.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.rewards import rewards


@pytest.fixture(params=["native", "qualified"])
def if_registry(request, tmp_path):
    if request.param == "native":
        guide = Path(__file__).resolve().parents[1] / "docs/miles/grpo.md"
        blocks = re.findall(r"```json\n(.*?)\n```", guide.read_text(), re.DOTALL)
        registry = json.loads(next(block for block in blocks if "ManifestIFVerifier" in block))
    else:
        module = os.environ.get("OPEN_INSTRUCT_TEST_QUALIFIED_IF_MODULE")
        if not module:
            pytest.skip("Run in the separately qualified image with OPEN_INSTRUCT_TEST_QUALIFIED_IF_MODULE")
        registry = {
            "ifeval": {
                "factory": "open_instruct.miles.rewards.task_verifiers.QualifiedIFVerifier",
                "config": {"module": module},
            }
        }
    path = tmp_path / "if-registry.json"
    path.write_text(json.dumps(registry))
    # Exercise construction from the documented entry, not a monkeypatched registry.
    rewards._registry(str(path))
    return path


CASES = [
    ("new:copy_span_idx", {"prompt_to_repeat": "hello", "n_start": 0, "n_end": 4}, "hello", "goodbye"),
    ("first_word:first_word_answer", {"first_word": "hello"}, "hello world", "goodbye world"),
    ("last_word:last_word_answer", {"last_word": "hello"}, "world hello", "world goodbye"),
    ("length_constraints:number_words", {"num_words": 2, "relation": "less than"}, "hello", "three separate words"),
    ("startend:quotation", {}, '"hello"', "hello"),
    ("detectable_format:json_format", {}, '{"hello": true}', "hello"),
    ("detectable_format:bigram_wrapping", {}, "<<hello world>>", "hello world"),
    ("startend:end_checker", {"end_phrase": "hello"}, "world hello", "world goodbye"),
]


def score(path, response, target, final_only, with_format=False):
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            reward_config=str(path), reward_zero_truncated=True, reward_final_answer_only=final_only
        )
    )
    specs = [{"name": "ifeval", "target": target}]
    if with_format:
        specs.append({"name": "r1_format", "target": ""})
    item = SimpleNamespace(
        response=response,
        tokens=[1, 2, 3],
        response_length=2,
        status="completed",
        prompt="IF test",
        metadata={"verifiers": specs},
    )
    result = asyncio.run(rewards.registered_reward(args, item))
    assert item.response == response and item.tokens == [1, 2, 3]
    return result


@pytest.mark.parametrize("instruction,kwargs,positive,negative", CASES)
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("wrapper", ["{}", "<answer>{}</answer>", " \n<|assistant|><answer> {} </answer> \n"])
def test_real_if_same_normalized_answer_in_both_modes(
    if_registry, instruction, kwargs, positive, negative, correct, wrapper
):
    target = {"instruction_id": [instruction], "kwargs": [kwargs]}
    response = "reasoning </think>" + wrapper.format(positive if correct else negative)
    assert [score(if_registry, response, target, final_only) for final_only in [False, True]] == [float(correct)] * 2


@pytest.mark.parametrize("final_only", [False, True])
def test_real_if_and_format_rewards_agree_on_tagged_copy(if_registry, final_only):
    registry = json.loads(if_registry.read_text())
    registry["r1_format"] = {"factory": "open_instruct.miles.rewards.task_verifiers.R1FormatVerifier"}
    path = if_registry.with_name("if-and-format.json")
    path.write_text(json.dumps(registry))
    target = {
        "instruction_id": ["new:copy_span_idx"],
        "kwargs": [{"prompt_to_repeat": "hello", "n_start": 0, "n_end": 4}],
    }
    assert score(path, "reasoning </think><answer>hello</answer>", target, final_only, with_format=True) == 2.0
