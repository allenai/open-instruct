"""Full-pool EMO across serving, MILES packing and Core replay (CPU tensors)."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from miles.backends.core_utils import data, models, moe_models, packing
from olmo_core.nn.moe import EmoRouterConfig
from olmo_core.nn.moe.v2 import replay
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_sglang import config as serving_config
from olmo_sglang import routing
from torch import nn

from open_instruct.miles.configuration.config import CoreConfig


def router_config():
    return MoERouterConfigV2(
        d_model=4,
        num_experts=8,
        top_k=2,
        normalize_expert_weights=1.0,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=4,
            eval_document_expert_pool=8,
            full_pool=True,
        ),
    )


def hf_config():
    return SimpleNamespace(
        model_type="olmo3moe",
        layer_types=["full_attention"],
        num_hidden_layers=1,
        use_head_qk_norm=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=4,
        emo_eval_document_expert_pool=8,
        emo_eos_token_id=0,
        emo_routing_mode="full_pool",
    )


def test_full_pool_serving_selections_replay_with_packed_eos_free_samples():
    torch.manual_seed(218)
    model = nn.Module()
    block = nn.Module()
    block.routed_experts_router = router_config().build()
    model.blocks = nn.ModuleDict({"0": block})
    router = block.routed_experts_router
    nn.init.normal_(router.weight, std=0.1)
    lengths = [3, 5]
    x = torch.randn(1, sum(lengths), 4, requires_grad=True)
    serving_config.validate_olmo3_moe_config(hf_config())
    logits = routing.fp32_router_logits(x.detach()[0], router.weight.detach().view(8, 4))
    weights, ids = routing.olmo3_moe_topk(
        x[0],
        logits,
        2,
        True,
        normalize_expert_weights=1.0,
        restore_weight_scale=False,
        original_num_experts_per_tok=None,
    )
    raw = dict(
        tokens=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])],
        total_lengths=lengths,
        response_lengths=[2, 4],
        loss_masks=[torch.ones(2), torch.ones(4)],
        rollout_routed_experts=[ids[:2, None], ids[3:7, None]],
    )
    samples = data.sample_batches(raw, 8)
    batch = packing.combine(samples, [0, 1])
    assert batch["doc_lens"].tolist() == [[3, 5]]
    routes = data.router_routes(model, batch)
    table = routes["blocks.0.routed_experts_router"]
    assert table[0, 2].tolist() == table[0, 7].tolist() == [0, 1]
    with mock.patch.object(router, "_pool_sizes", side_effect=AssertionError("document pool sampled")):
        with replay.replay_routes(model, routes):
            actual, selected, _, _ = router(x, False)
            torch.testing.assert_close(selected, table)
            torch.testing.assert_close(actual[0, [0, 1, 3, 4, 5, 6]], weights[[0, 1, 3, 4, 5, 6]])
            (actual[0, :2] * torch.tensor([1.0, 3.0])).sum().backward()
        assert not router.requires_segment_ids
        assert x.grad[:, :2].abs().sum() > 0
        assert x.grad[:, 2:].count_nonzero() == 0
        assert router.weight.grad.abs().sum() > 0
        fresh = router(x.detach(), False)
        changed = x.detach().clone()
        changed[:, 3:] *= -100
        isolated = router(changed, False)
        torch.testing.assert_close(fresh[0][:, :3], isolated[0][:, :3])
        torch.testing.assert_close(fresh[1][:, :3], isolated[1][:, :3])
        for start, stop, sample in ((0, 3, samples[0]), (3, 8, samples[1])):
            with replay.replay_routes(model, data.router_routes(model, sample)):
                separate = router(x.detach()[:, start:stop], False)
            torch.testing.assert_close(separate[0], actual[:, start:stop])


@pytest.mark.parametrize("failure", [None, "replay", "aux", "z", "counts", "native", "custom", "gating"])
def test_trainer_rejects_unqualified_emo_before_model_build(failure):
    router = router_config()
    config = SimpleNamespace(block=SimpleNamespace(routed_experts_router=router), block_overrides={})
    hf = hf_config()
    args = SimpleNamespace(
        use_rollout_routing_replay=True, olmo_core=CoreConfig(router_aux_loss_weight=0, router_z_loss_weight=0)
    )
    if failure == "replay":
        args.use_rollout_routing_replay = False
    elif failure == "aux":
        args.olmo_core = replace(args.olmo_core, router_aux_loss_weight=0.01)
    elif failure == "z":
        args.olmo_core = replace(args.olmo_core, router_z_loss_weight=1e-5)
    elif failure == "counts":
        args.olmo_core = replace(args.olmo_core, router_aux_count_source="router_selected")
    elif failure == "native":
        hf.emo_routing_mode = None
    elif failure == "custom":
        config.block_overrides[1] = deepcopy(config.block)
        config.block_overrides[1].routed_experts_router.emo.full_pool = False
    elif failure == "gating":
        router.gating_function = "topk_softmax"
    if failure:
        with pytest.raises(ValueError, match="EMO"):
            moe_models.validate_emo_training(config, hf, args)
    else:
        moe_models.validate_emo_training(config, hf, args)


def test_actual_builder_validates_before_cuda_or_weights(tmp_path):
    from_fixture = {
        **vars(hf_config()),
        "vocab_size": 32,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "moe_intermediate_size": 16,
        "normalize_expert_weights": 1.0,
        "dense_layers_indices": [],
        "eos_token_id": 0,
    }
    # Use the actual registered HF class to exercise metadata propagation.
    moe_models.register_hf_classes()
    models.transformers.AutoConfig.for_model(**from_fixture).save_pretrained(tmp_path)
    args = SimpleNamespace(
        hf_checkpoint=str(tmp_path),
        seed=17,
        use_rollout_routing_replay=False,
        olmo_core=CoreConfig(attention_backend="torch", router_aux_loss_weight=0, router_z_loss_weight=0),
    )
    with (
        mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA touched")),
        mock.patch.object(
            models.transformers.AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("weights loaded")
        ),
        pytest.raises(ValueError, match="use_rollout_routing_replay"),
    ):
        models.build_train_module(args)
