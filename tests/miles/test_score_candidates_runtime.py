"""Compile the local score options through the real MILES parser."""

import contextlib
import io
import sys

import pytest
from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.utils import arguments

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("mode,transport", [("topk", "nested"), ("topk", "flat"), ("topk", "array"), ("mc", "array")])
def test_score_options_reach_native_parser(monkeypatch, width, mode, transport):
    monkeypatch.setattr(sys, "argv", ["test"])
    with contextlib.redirect_stderr(io.StringIO()):
        parser = fsdp_arguments.build_fsdp_parser(arguments.get_miles_extra_args_provider())
    parser.add_argument("--olmo-core-config", required=True)
    requested = {
        "hf_checkpoint": "model",
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 4,
        "global_batch_size": 4,
        "rollout_top_logprobs_num": width,
        "rollout_score_mode": mode,
        "rollout_score_transport": transport,
        "loss_type": "policy_loss" if mode == "mc" else "score_centering",
    }
    parsed = parser.parse_args(RunConfig(CoreConfig(), requested).arguments())
    for name, value in requested.items():
        assert getattr(parsed, name) == value
    assert parsed.rollout_top_k == -1
