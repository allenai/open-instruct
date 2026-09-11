"""Round-trip the public compiler through the actual pinned MILES/SGLang parser."""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest
from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.utils import arguments

from open_instruct.miles import arguments as core_arguments
from open_instruct.miles import options
from open_instruct.miles.config import CoreConfig, RunConfig


@pytest.fixture(scope="module")
def parser():
    argv = sys.argv
    try:
        sys.argv = ["test"]
        with contextlib.redirect_stderr(io.StringIO()):
            result = fsdp_arguments.build_fsdp_parser(arguments.get_miles_extra_args_provider())
        result.add_argument("--olmo-core-config", required=True)
        return result
    finally:
        sys.argv = argv


def test_schema_matches_installed_runtime(parser):
    recorded = json.loads(Path(options.__file__).with_name("options.json").read_text())
    assert options.describe_parser(parser) == recorded["options"]


def test_every_boolean_round_trips(parser):
    count = 0
    for name, record in options.option_index().items():
        if name != record["dest"] or record["kind"] not in ("boolean", "switch"):
            continue
        for value in (False, True):
            if record["kind"] == "switch" and value not in (record["default"], record["const"]):
                continue
            encoded = options.encode_options({name: value})
            parsed = parser.parse_args(["--olmo-core-config", "{}", "--rollout-batch-size", "1", *encoded])
            assert getattr(parsed, name) == value, (name, value, encoded)
            count += 1
    assert count > 100


def test_mixed_native_controls_round_trip(parser):
    requested = {
        "train_env_vars": {"NCCL_DEBUG": "WARN"},
        "eval_prompt_data": ["gsm8k", "/data/heldout.jsonl"],
        "rollout_stop_token_ids": [1, 2],
        "grpo_std_normalization": False,
        "rollout_global_dataset": False,
        "use_wandb": False,
        "colocate": False,
        "sglang_cuda_graph_backend_decode": "full",
        "sglang_cuda_graph_max_bs_decode": 64,
        "sglang_disable_radix_cache": False,
        "sglang_max_running_requests": 64,
        "sglang_mem_fraction_static": 0.85,
        "sglang_speculative_algorithm": "NGRAM",
        "async_data_buffer_capacity_factor": 2.0,
        "async_unused_samples_handler": "retry",
        "lr": 1e-6,
        "eps_clip_high": 0.28,
    }
    parsed = parser.parse_args(
        ["--olmo-core-config", "{}", "--rollout-batch-size", "1", *options.encode_options(requested)]
    )
    for key, value in requested.items():
        assert getattr(parsed, key) == value


def test_serving_json_and_legacy_aliases(parser):
    requested = {
        "sglang_cuda_graph_config": {"decode": {"backend": "full", "max_bs": 4}},
        "sglang_mamba_scheduler_strategy": "extra_buffer",
        "sglang_disable_piecewise_cuda_graph": True,
        "sglang_disable_cuda_graph": False,
        "sglang_json_model_override_args": {"max_position_embeddings": 8192},
    }
    parsed = parser.parse_args(
        ["--olmo-core-config", "{}", "--rollout-batch-size", "1", *options.encode_options(requested)]
    )
    assert parsed.sglang_cuda_graph_config == requested["sglang_cuda_graph_config"]
    assert parsed.sglang_mamba_radix_cache_strategy == "extra_buffer"
    assert parsed.sglang_cuda_graph_backend_prefill == "disabled"
    assert not parsed.sglang_disable_cuda_graph
    assert json.loads(parsed.sglang_json_model_override_args) == requested["sglang_json_model_override_args"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--async-save"],
        ["--no-save-optim"],
        ["--reset-optimizer-states"],
        ["--override-lr-scheduler", "--no-use-checkpoint-lr-scheduler"],
        ["--no-use-checkpoint-lr-scheduler"],
        ["--disable-compute-advantages-and-returns"],
        ["--skip-actor-forward-only"],
        ["--keep-old-actor"],
        ["--dp-replicate-size", "2"],
        ["--deterministic-mode"],
        ["--lora-train-only"],
        ["--lora-rank", "8"],
        ["--save-hf", "/data/hf/{rollout_id}"],
        ["--debug-disable-optimizer"],
        ["--update-weight-transfer-mode", "disk-delta"],
        ["--data-source-path", "custom.Source"],
        ["--max-weight-staleness", "3"],
    ],
)
def test_direct_native_cli_checks_backend_contract(monkeypatch, extra):
    config = RunConfig(
        CoreConfig(),
        {"hf_checkpoint": "model", "rollout_batch_size": 1, "n_samples_per_prompt": 4, "global_batch_size": 4},
    )
    monkeypatch.setattr(sys, "argv", ["test", *config.arguments(), *extra])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(ValueError):
        core_arguments.load_core_args(arguments.get_miles_extra_args_provider())


def test_async_lag_uses_core_optimizer_steps(monkeypatch):
    config = RunConfig(
        CoreConfig(max_policy_lag=2),
        {
            "hf_checkpoint": "model",
            "rollout_batch_size": 1,
            "n_samples_per_prompt": 4,
            "global_batch_size": 4,
            "fully_async": True,
        },
    )
    monkeypatch.setattr(sys, "argv", ["test", *config.arguments()])
    with contextlib.redirect_stderr(io.StringIO()):
        parsed = core_arguments.load_core_args(arguments.get_miles_extra_args_provider())
    assert parsed.max_weight_staleness == 2
    assert parsed.custom_async_data_buffer_path == "open_instruct.miles.async_buffer.HomogeneousPolicyDataBuffer"
