"""Round-trip the experimental Core IcePop settings through the pinned runtime."""

import contextlib
import io
import sys

import pytest
from miles.backends.core_utils.icepop_config import ICEPOP_LOSS, ICEPOP_SCMC_LOSS, validate_icepop_args
from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.utils import arguments

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


@pytest.mark.parametrize("width", [0, 32, 64, 128])
def test_icepop_configuration_reaches_native_loss(monkeypatch, width):
    monkeypatch.setattr(sys, "argv", ["test"])
    with contextlib.redirect_stderr(io.StringIO()):
        parser = fsdp_arguments.build_fsdp_parser(arguments.get_miles_extra_args_provider())
    parser.add_argument("--olmo-core-config", required=True)
    requested = {
        "hf_checkpoint": "model",
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 4,
        "global_batch_size": 4,
        "loss_type": "custom_loss",
        "custom_loss_function_path": ICEPOP_SCMC_LOSS if width else ICEPOP_LOSS,
        "score_centering_is": "mis",
        "score_centering_mis_low": 0.2,
        "score_centering_mis_high": 5.0,
        "use_tis": False,
        "use_rollout_logprobs": True,
        "rollout_top_logprobs_num": width,
        "rollout_score_mode": "mc" if width else "topk",
        "rollout_score_transport": "array" if width else "nested",
        "use_miles_router": True,
    }
    parsed = parser.parse_args(RunConfig(CoreConfig(), requested).arguments())
    for name, value in requested.items():
        assert getattr(parsed, name) == value
    validate_icepop_args(parsed)
    assert get_loss_function(parsed).__name__ == requested["custom_loss_function_path"].rsplit(".", 1)[1]
