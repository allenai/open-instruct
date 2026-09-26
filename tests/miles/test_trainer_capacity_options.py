"""Check opt-in options reach native MoE constructors and reject dense misuse."""

from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.training import moe_models, standard_models


@pytest.mark.parametrize("enabled", [False, True])
def test_native_optimizer_and_reduction_options(enabled, monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(moe_models, "HFInitializedMoETrainModule", constructor)
    args = SimpleNamespace(
        olmo_core=CoreConfig(compile_optimizer=enabled, use_reduce_scatter=enabled, expert_parallel_size=2),
        clip_grad=1.0,
    )
    module = moe_models.build_train_module(args, common={}, optim={"lr": 1e-6}, hf_config=None, hf_state={})
    assert captured["optim"].compile is enabled
    assert captured["dp_config"].use_reduce_scatter is enabled
    assert captured["ep_config"].degree == 2
    assert module._miles_checkpoint_options == args.olmo_core.checkpoint_save_options()


@pytest.mark.parametrize("option", ["compile_optimizer", "use_reduce_scatter"])
def test_dense_backend_rejects_moe_specific_options(option):
    with pytest.raises(ValueError, match="require the Core MoE trainer"):
        standard_models.validate_training_options(SimpleNamespace(olmo_core=CoreConfig(**{option: True})))
