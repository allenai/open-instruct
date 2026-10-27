"""Native serving IDs differ from trainer IDs only by declared image expansion."""

import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch
from miles.utils import types

from miles.rollout import base_types
from open_instruct.miles.datasets import vision_rollout


@pytest.fixture
def generate_request(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(b"exact-image-bytes")
    expanded = [0, 7, 3, 3, 8, 2]
    prepared = SimpleNamespace(
        patches=torch.ones(1, 1, 4, 12), pooling=torch.tensor([[[0, 1], [2, 3]]]), offsets=[(2, 3)]
    )
    sample = types.Sample(
        prompt="<|image|> A or B?",
        metadata={
            "vision": {"images": [{"path": str(path)}], "prepared": {"placeholder_token_ids": [0, 1, 2]}},
            "prepared_sample_id": "sample-0",
            "source_image_sha256": ["a" * 64],
            "pair_id": "pair-0",
        },
    )
    args = SimpleNamespace(
        hf_checkpoint="checkpoint",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_max_prompt_len=8,
        rollout_max_response_len=2,
        rollout_max_context_len=8,
        sglang_router_ip="localhost",
        sglang_router_port=123,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
    )
    monkeypatch.setattr(
        vision_rollout, "processor", lambda _: SimpleNamespace(token_ids=SimpleNamespace(im_patch_id=3))
    )
    monkeypatch.setattr(vision_rollout.vision, "process_row", lambda *args: (expanded, [prepared]))
    return base_types.GenerateFnInput(SimpleNamespace(args=args), sample, {}, False)


def output(prompt_tokens=6):
    return {
        "text": "A",
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "output_token_logprobs": [[-0.7, 9, "A"]],
            "finish_reason": {"type": "stop"},
            "weight_version": "2",
        },
    }


def test_native_request_expanded_training_tokens_and_policy_provenance(generate_request, monkeypatch):
    async def post(url, payload, headers):
        assert url.endswith(":123/generate")
        assert payload["input_ids"] == [0, 1, 2]
        assert base64.b64decode(payload["image_data"][0].split(",", 1)[1]) == b"exact-image-bytes"
        assert payload["sampling_params"]["max_new_tokens"] == 2
        return output()

    monkeypatch.setattr(vision_rollout.http_utils, "post", post)
    result = asyncio.run(vision_rollout.generate(generate_request))
    sample = result.samples
    assert sample.tokens == [0, 7, 3, 3, 8, 2, 9]
    assert sample.response_length == 1 and sample.rollout_log_probs == [-0.7]
    assert sample.status == types.Sample.Status.COMPLETED
    assert sample.weight_versions[0].to_dicts() == [{"version": "2", "abs_start": 6, "abs_end": 7}]
    assert sample.multimodal_train_inputs["image_positions"].tolist() == [2, 3]
    assert sample.train_metadata["pair_id"] == "pair-0"


def test_serving_expansion_disagreement_is_fatal(generate_request, monkeypatch):
    async def post(*args, **kwargs):
        return output(prompt_tokens=3)

    monkeypatch.setattr(vision_rollout.http_utils, "post", post)
    with pytest.raises(ValueError, match="disagree"):
        asyncio.run(vision_rollout.generate(generate_request))


@pytest.mark.parametrize("damage", ["context", "partial", "unprepared", "replay"])
def test_unsupported_input_fails_before_request(generate_request, monkeypatch, damage):
    async def post(*args, **kwargs):
        raise AssertionError("invalid generate_request reached serving")

    monkeypatch.setattr(vision_rollout.http_utils, "post", post)
    if damage == "context":
        generate_request.args.rollout_max_context_len = 7
    elif damage == "partial":
        generate_request.sample.response = "unfinished"
    elif damage == "unprepared":
        generate_request.sample.metadata["vision"].pop("prepared")
    else:
        generate_request.args.use_rollout_routing_replay = True
    with pytest.raises(ValueError):
        asyncio.run(vision_rollout.generate(generate_request))
