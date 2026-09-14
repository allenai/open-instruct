"""Budget and payload regressions for the long-input GPU exercise."""

import pytest
from scripts.miles import launch_readiness_lengths, readiness_lengths


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, **kwargs):
        return list(text.encode())


@pytest.mark.parametrize("length", [16384 - 256, 32768 - 256, 65536 - 256])
def test_budget_preserves_template_and_needle(length):
    tokens = readiness_lengths.prompt_tokens(CharacterTokenizer(), length, 2)
    assert len(tokens) == length
    prompt = bytes(tokens).decode()
    assert prompt.startswith("<user>")
    assert prompt.endswith("</user><assistant>")
    assert "The secret number for this document is 733." in prompt
    assert "Return only the secret number." in prompt
    assert "<FILLER>" not in prompt


def test_impossible_budget_rejected():
    with pytest.raises(ValueError, match="budget"):
        readiness_lengths.prompt_tokens(CharacterTokenizer(), 4, 0)


def test_launch_is_single_gpu_and_pins_script():
    spec = launch_readiness_lengths.specification("immutable-image", b"print('probe')")
    task = spec["tasks"][0]
    assert task["resources"]["gpuCount"] == 1
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["image"]["beaker"] == "immutable-image"
    assert "python /output/lengths.py" in task["arguments"][0]
