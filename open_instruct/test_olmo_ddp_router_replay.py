import contextlib
from types import SimpleNamespace

import pytest
import torch
import transformers
from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName

from open_instruct import grpo_utils, olmo_core_utils, vllm_utils


class _ReplayBlock(torch.nn.Module):
    def __init__(self, *, routed: bool):
        super().__init__()
        self.routed_experts_router = torch.nn.Identity() if routed else None


class _MixedReplayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleDict(
            {"0": _ReplayBlock(routed=False), "1": _ReplayBlock(routed=True), "2": _ReplayBlock(routed=True)}
        )


def test_olmo_ddp_ep_config_accepts_divisible_learner_world():
    config = grpo_utils.GRPOExperimentConfig(
        olmo_core_train_module="ddp", olmo_core_ep_degree=2, num_learners_per_node=[4]
    )
    assert config.olmo_core_ep_degree == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"olmo_core_ep_degree": 3, "num_learners_per_node": [4]}, "evenly divide"),
        ({"cp_degree": 2}, "Context parallelism"),
        ({"sequence_parallel_size": 2}, "Sequence parallelism"),
        ({"ref_policy_update_freq": 10}, "reference-policy updates"),
        ({"gather_whole_model": False}, "gather_whole_model"),
    ],
)
def test_olmo_ddp_stage_one_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        grpo_utils.GRPOExperimentConfig(olmo_core_train_module="ddp", **kwargs)


class _RemoteMethod:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.value


def test_prepared_hf_state_ipc_sync(monkeypatch):
    engine = SimpleNamespace(sleep=_RemoteMethod("slept"), set_model_step=_RemoteMethod("stepped"))
    sent = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(vllm_utils.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        vllm_utils.IPCWeightTransferEngine,
        "trainer_send_weights",
        lambda iterator, trainer_args: sent.extend(iterator),
    )
    weights = {"model.embed_tokens.weight": torch.arange(4).reshape(2, 2)}

    refs = vllm_utils.broadcast_prepared_weights_to_vllm(
        weights=weights, vllm_engines=[engine], model_update_group=None, model_step=7
    )

    assert refs == ["stepped"]
    assert engine.set_model_step.calls == [((7,), {})]
    assert [name for name, _ in sent] == list(weights)
    torch.testing.assert_close(sent[0][1], weights["model.embed_tokens.weight"])


def test_olmo_ddp_factory_builds_qwen3_moe_config():
    hf_config = transformers.Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        max_position_embeddings=32,
    )

    model_config = olmo_core_utils.build_olmo_ddp_model_config_from_hf_config(
        hf_config, dtype=DType.float32, attention_backend=AttentionBackendName.torch
    )

    assert model_config.n_layers == 2
    assert model_config.block.routed_experts_router.top_k == 2
    assert model_config.block.routed_experts_router.lb_loss_weight is None
    assert model_config.block.routed_experts_router.normalize_expert_weights is None


def test_olmo_ddp_factory_rejects_unsupported_checkpoint_type():
    with pytest.raises(ValueError, match="gpt2"):
        olmo_core_utils.build_olmo_ddp_model_config_from_hf_config(
            transformers.GPT2Config(), dtype=DType.float32, attention_backend=AttentionBackendName.torch
        )


def test_replay_router_context_removes_dense_layer_slots(monkeypatch):
    captured = []

    @contextlib.contextmanager
    def capture_replay(_model, per_layer_indices):
        captured.extend(per_layer_indices)
        yield

    monkeypatch.setattr(olmo_core_utils.olmo_moe, "replay_routing", capture_replay)
    routes = torch.arange(3).reshape(1, 1, 3, 1)

    with olmo_core_utils.replay_router_context(_MixedReplayModel(), routes):
        pass

    assert len(captured) == 2
    torch.testing.assert_close(captured[0], routes[:, :, 1, :])
    torch.testing.assert_close(captured[1], routes[:, :, 2, :])


def test_replay_router_context_accepts_router_only_layer_axis(monkeypatch):
    captured = []

    @contextlib.contextmanager
    def capture_replay(_model, per_layer_indices):
        captured.extend(per_layer_indices)
        yield

    monkeypatch.setattr(olmo_core_utils.olmo_moe, "replay_routing", capture_replay)
    routes = torch.arange(2).reshape(1, 1, 2, 1)

    with olmo_core_utils.replay_router_context(_MixedReplayModel(), routes):
        pass

    assert len(captured) == 2
    torch.testing.assert_close(captured[0], routes[:, :, 0, :])
    torch.testing.assert_close(captured[1], routes[:, :, 1, :])


def test_streamed_hf_state_ipc_sync(monkeypatch):
    engine = SimpleNamespace(sleep=_RemoteMethod("slept"), set_model_step=_RemoteMethod("stepped"))
    sent = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(vllm_utils.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        vllm_utils.IPCWeightTransferEngine,
        "trainer_send_weights",
        lambda iterator, trainer_args: sent.extend(iterator),
    )
    weight = torch.arange(4).reshape(2, 2)
    metadata = [("model.embed_tokens.weight", weight.dtype, tuple(weight.shape))]

    refs = vllm_utils.broadcast_streamed_weights_to_vllm(
        metadata=metadata,
        weights=iter([("model.embed_tokens.weight", weight)]),
        vllm_engines=[engine],
        model_update_group=None,
        model_step=9,
    )

    assert refs == ["stepped"]
    assert engine.set_model_step.calls == [((9,), {})]
    assert [name for name, _ in sent] == ["model.embed_tokens.weight"]
    torch.testing.assert_close(sent[0][1], weight)


def test_streamed_hf_state_nonzero_rank_drains_collective_iterator(monkeypatch):
    drained = []

    def weights():
        drained.append(True)
        yield "unused", torch.ones(1)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    refs = vllm_utils.broadcast_streamed_weights_to_vllm(
        metadata=[], weights=weights(), vllm_engines=[], model_update_group=object(), model_step=1
    )

    assert refs == []
    assert drained == [True]


def test_cpu_staged_hf_state_nccl_sync(monkeypatch):
    engine = SimpleNamespace(sleep=_RemoteMethod("slept"), update_weights=_RemoteMethod("updated"))
    sent = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(vllm_utils.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        vllm_utils.NCCLWeightTransferEngine,
        "trainer_send_weights",
        lambda iterator, trainer_args: sent.extend(iterator),
    )
    weights = {"model.embed_tokens.weight": torch.arange(4).reshape(2, 2), "model.norm.weight": torch.arange(2)}

    refs = vllm_utils.broadcast_cpu_staged_weights_to_vllm(
        weights=weights, vllm_engines=[engine], model_update_group=object(), model_step=11, staging_device="cpu"
    )

    assert refs == ["updated"]
    update_request = engine.update_weights.calls[0][0][0]
    assert update_request["update_info"] == {
        "names": list(weights),
        "dtype_names": ["int64", "int64"],
        "shapes": [[2, 2], [2]],
        "packed": False,
    }
    assert [name for name, _ in sent] == list(weights)
    for (name, tensor), expected in zip(sent, weights.values()):
        assert tensor.device.type == "cpu", name
        torch.testing.assert_close(tensor, expected)
