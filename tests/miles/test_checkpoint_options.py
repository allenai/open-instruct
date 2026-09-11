"""Configuration controls reach native saves without changing dense-trainer behavior."""

from types import SimpleNamespace
from unittest import mock

import pytest

from open_instruct.miles import models, standard_models
from open_instruct.miles.config import CoreConfig


@pytest.mark.parametrize("field", ["checkpoint_thread_count", "checkpoint_process_count"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_worker_counts_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_profile",
        "checkpoint_compact_storage",
        "checkpoint_dedup_save_to_lowest_rank",
        "checkpoint_constant_memory_planning",
    ],
)
def test_checkpoint_switches_are_booleans(field):
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: "false"})


def test_default_checkpoint_policy_remains_legacy():
    assert CoreConfig().checkpoint_save_options() == {
        "profile": False,
        "compact_storage": False,
        "dedup_save_to_lowest_rank": True,
        "constant_memory_planning": False,
    }


def test_native_save_receives_policy_and_returns_measurements(tmp_path):
    config = CoreConfig(
        checkpoint_profile=True,
        checkpoint_thread_count=4,
        checkpoint_process_count=2,
        checkpoint_compact_storage=True,
        checkpoint_dedup_save_to_lowest_rank=False,
        checkpoint_constant_memory_planning=True,
    )
    measurements = {"total_seconds": 100.0}
    module = SimpleNamespace(
        _miles_model_backend="moe",
        _miles_checkpoint_options=config.checkpoint_save_options(),
        save_state_dict_direct=mock.Mock(return_value=measurements),
    )
    assert models.save_native(module, tmp_path) is measurements
    module.save_state_dict_direct.assert_called_once_with(str(tmp_path), **config.checkpoint_save_options())


def test_dense_backend_rejects_unqualified_checkpoint_policy():
    with pytest.raises(ValueError, match="Checkpoint writer overrides"):
        standard_models.validate_training_options(SimpleNamespace(olmo_core=CoreConfig(checkpoint_profile=True)))
    standard_models.validate_training_options(SimpleNamespace(olmo_core=CoreConfig()))


@pytest.mark.parametrize("enabled", [False, True])
def test_resume_uses_only_qualified_read_controls(tmp_path, enabled):
    config = CoreConfig(
        checkpoint_profile=enabled,
        checkpoint_constant_memory_planning=enabled,
        checkpoint_process_count=2,
        checkpoint_thread_count=4,
        checkpoint_compact_storage=True,
    )
    module = SimpleNamespace(
        _miles_model_backend="moe",
        _miles_checkpoint_options=config.checkpoint_save_options(),
        load_state_dict_direct=mock.Mock(),
    )
    models.load_native(module, tmp_path)
    expected = {"profile": True, "constant_memory_planning": True} if enabled else {}
    module.load_state_dict_direct.assert_called_once_with(str(tmp_path), load_optim_state=True, **expected)
