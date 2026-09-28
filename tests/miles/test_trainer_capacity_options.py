"""Check opt-in options reach native MoE constructors and reject dense misuse."""

import inspect
from types import SimpleNamespace

import pytest
from miles.backends.core_utils import moe_models, standard_models
from olmo_core.train.train_module import transformer as train_transformer

from open_instruct.miles.configuration.config import CoreConfig


@pytest.mark.parametrize("enabled", [False, True])
def test_native_optimizer_and_reduction_options(enabled, monkeypatch):
    captured = {}

    signature = inspect.signature(train_transformer.OLMoDDPTrainModule.__init__)

    def constructor(**kwargs):
        native = {k: v for k, v in kwargs.items() if k not in {"hf_config", "hf_state", "startup_args"}}
        signature.bind(None, **native)
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(moe_models, "HFInitializedMoETrainModule", constructor)
    args = SimpleNamespace(
        olmo_core=CoreConfig(compile_optimizer=enabled, use_reduce_scatter=enabled, expert_parallel_size=2),
        clip_grad=1.0,
    )
    common = dict(model=object(), rank_microbatch_size=16, max_sequence_length=16, max_grad_norm=args.clip_grad)
    module = moe_models.build_train_module(args, common=common, optim={"lr": 1e-6}, hf_config=None, hf_state={})
    assert "max_grad_norm" not in captured
    assert captured["optim"].max_grad_norm == args.clip_grad
    assert common["max_grad_norm"] == args.clip_grad
    assert captured["optim"].compile is enabled
    assert captured["dp_config"].use_reduce_scatter is enabled
    assert captured["ep_config"].degree == 2
    assert module._miles_checkpoint_options == args.olmo_core.checkpoint_save_options()


@pytest.mark.parametrize("option", ["compile_optimizer", "use_reduce_scatter"])
def test_dense_backend_rejects_moe_specific_options(option):
    with pytest.raises(ValueError, match="require the Core MoE trainer"):
        standard_models.validate_training_options(SimpleNamespace(olmo_core=CoreConfig(**{option: True})))
