"""Check that resumed rollouts receive the restored learner policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct import grpo_callbacks


@pytest.mark.parametrize("checkpoint_loaded", [False, True])
def test_publish_restored_policy_before_training(monkeypatch, checkpoint_loaded):
    model = object()
    mapper = Mock()
    callback = grpo_callbacks.VLLMWeightSyncCallback(
        vllm_engines=[Mock()], actor_manager=Mock(), model_update_group=object(), name_mapper=mapper
    )
    callback.trainer = SimpleNamespace(
        checkpoint_loaded=checkpoint_loaded,
        global_step=700 if checkpoint_loaded else 0,
        train_module=SimpleNamespace(model=model),
        record_metric=Mock(),
    )
    broadcast = Mock(return_value=["weight-transfer"])
    sync = Mock(return_value=({}, None))
    monkeypatch.setattr(grpo_callbacks.vllm_utils, "broadcast_weights_to_vllm", broadcast)
    monkeypatch.setattr(grpo_callbacks.grpo_utils, "perform_weight_sync", sync)
    monkeypatch.setattr(grpo_callbacks.torch.cuda, "empty_cache", Mock())

    callback.pre_train()

    if checkpoint_loaded:
        broadcast.assert_called_once_with(
            model=model,
            vllm_engines=callback.vllm_engines,
            model_update_group=callback.model_update_group,
            model_step=700,
            name_mapper=mapper,
        )
        sync.assert_called_once_with(
            ["weight-transfer"], callback.vllm_engines, callback.actor_manager, inflight_updates=True
        )
    else:
        broadcast.assert_not_called()
        sync.assert_not_called()
