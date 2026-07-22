from types import SimpleNamespace

import pytest
import torch

from open_instruct import grpo_utils, vllm_utils


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
