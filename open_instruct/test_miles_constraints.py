"""Shared constraints must keep validation, native entry points and docs aligned.
Exercise real rejected and accepted configurations, including the distinction
between invalid combinations, missing support and unqualified runtime settings.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import generate_docs

from open_instruct.miles.configuration import constraints
from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "configs/miles/examples/small.toml"


@pytest.mark.parametrize(
    "options,category,detail",
    [
        ({"actor_num_gpus_per_node": 3}, constraints.Category.INVALID, "positive multiple"),
        ({"async_save": True}, constraints.Category.NOT_IMPLEMENTED, "async_save"),
        ({"offload_train": True}, constraints.Category.NOT_VALIDATED, "offload_train=false"),
        ({"gradient_checkpointing": False}, constraints.Category.INVALID, "core.activation_checkpointing"),
    ],
)
def test_backend_rejections_explain_the_category(options, category, detail):
    config = RunConfig(CoreConfig(), {"hf_checkpoint": "fixture", "global_batch_size": 8, **options})
    with pytest.raises(InputError) as caught:
        config.validate()
    message = str(caught.value)
    assert message.startswith(category.value + ":")
    assert detail in message
    assert ("contribute support" in message) == (category == constraints.Category.NOT_IMPLEMENTED)
    assert ("share the results" in message) == (category == constraints.Category.NOT_VALIDATED)


def test_fixed_rule_changes_reach_validation_and_documentation(monkeypatch):
    config = RunConfig(CoreConfig(), {"hf_checkpoint": "fixture", "global_batch_size": 8, "async_save": False})
    config.validate()
    monkeypatch.setitem(constraints.REQUIRED_VALUES, "async_save", True)
    with pytest.raises(InputError, match="requires miles.async_save=True"):
        config.validate()
    dataclasses.replace(config, miles=config.miles | {"async_save": True}).validate()
    assert "true" in generate_docs.source_constraints()["async_save"]


def test_native_collection_preserves_explicit_only_options_and_false_values():
    args = SimpleNamespace(
        hf_checkpoint="fixture",
        global_batch_size=8,
        fp16=False,
        offload_train=False,
        gradient_checkpointing=False,
        max_weight_staleness=1,
        unrelated_native_option="ignored",
        save_hf=None,
    )
    fields = constraints.native_values(args, set())
    assert fields == {"hf_checkpoint": "fixture", "global_batch_size": 8, "fp16": False, "offload_train": False}
    RunConfig(CoreConfig(), fields).validate()
    fields = constraints.native_values(args, {"--gradient-checkpointing"})
    with pytest.raises(InputError, match="core.activation_checkpointing"):
        RunConfig(CoreConfig(), fields).validate()


@pytest.mark.parametrize(
    "override,category",
    [
        ('runtime.output_dir="/tmp/other"', constraints.Category.INVALID),
        ("runtime.rollout_stage_timeout=60", constraints.Category.NOT_IMPLEMENTED),
        ("miles.offload_train=true", constraints.Category.NOT_VALIDATED),
    ],
)
def test_structured_cli_constraints_still_stop_before_training(override, category):
    with pytest.raises(InputError, match=category.value):
        RunSpec.load(SMALL, [override])
    result = subprocess.run(
        [sys.executable, "-m", "open_instruct.miles", "plan", str(SMALL), "--set", override],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert category.value in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "change,category",
    [
        ({"fully_async": False}, constraints.Category.INVALID),
        ({"rollout_num_gpus_per_engine": 2}, constraints.Category.NOT_IMPLEMENTED),
        ({"rollout_temperature": 0.7}, constraints.Category.NOT_VALIDATED),
    ],
)
def test_refresh_preserves_categories_from_shared_publication_checks(change, category):
    config = RunConfig(
        CoreConfig(publication_mode="refresh", max_policy_lag=2),
        {
            "hf_checkpoint": "fixture",
            "global_batch_size": 4,
            "rollout_batch_size": 2,
            "n_samples_per_prompt": 2,
            "fully_async": True,
            "use_miles_router": True,
            "use_tis": True,
            "sglang_cuda_graph_backend_decode": "disabled",
            "sglang_cuda_graph_backend_prefill": "disabled",
            **change,
        },
    )
    with pytest.raises(InputError) as caught:
        config.validate()
    message = str(caught.value)
    assert message.startswith(category.value + ":")
    assert message.count(category.value) == 1
    assert "refresh" in message and "engine_drain" not in message
