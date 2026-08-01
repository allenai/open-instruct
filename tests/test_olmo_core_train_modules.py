"""Tests for OLMo-core GRPO train-module integration."""

import torch

from open_instruct import olmo_core_train_modules


class FakeOLMoDDPModel:
    def __init__(self) -> None:
        self.init_kwargs = None

    def init_weights(self, **kwargs) -> None:
        self.init_kwargs = kwargs


def test_hf_initializing_olmo_ddp_loads_state_during_model_initialization(monkeypatch):
    train_module = object.__new__(olmo_core_train_modules.HFInitializingOLMoDDPTrainModule)
    hf_config = object()
    hf_state = {"weight": torch.ones(1)}
    train_module._initial_hf_config = hf_config
    train_module._initial_hf_state = hf_state
    train_module.device = torch.device("cpu")
    train_module.world_mesh = object()
    model = FakeOLMoDDPModel()
    loaded = []

    monkeypatch.setattr(
        olmo_core_train_modules,
        "load_olmo_ddp_hf_state",
        lambda loaded_model, loaded_config, loaded_state: loaded.append((loaded_model, loaded_config, loaded_state)),
    )

    train_module.init_model_part_weights(model, model_part_idx=0, max_sequence_length=128, rank_microbatch_size=256)

    assert model.init_kwargs == {
        "max_seq_len": 128,
        "max_local_microbatch_size": 256,
        "device": torch.device("cpu"),
        "world_mesh": train_module.world_mesh,
        "model_part_idx": 0,
        "initialize_parameters": False,
    }
    assert loaded == [(model, hf_config, hf_state)]
    assert train_module._initial_hf_config is None
    assert train_module._initial_hf_state is None
