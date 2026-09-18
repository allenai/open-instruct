"""Expose the Core-owned count policy without changing other model options."""

from types import SimpleNamespace

import pytest

from open_instruct.miles import moe_models
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.errors import InputError


@pytest.mark.parametrize("source", ["dispatch", "current"])
def test_count_source_reaches_all_router_blocks(source):
    blocks = [
        SimpleNamespace(routed_experts_router=SimpleNamespace(lb_loss_count_source="dispatch"), routed_experts=None)
        for _ in range(3)
    ]
    config = SimpleNamespace(block={"base": blocks[0], "other": blocks[1]}, block_overrides={2: blocks[2]})
    options = CoreConfig(router_aux_count_source=source)
    moe_models.prepare_model_config(config, SimpleNamespace(layer_types=[]), options)
    assert all(block.routed_experts_router.lb_loss_count_source == source for block in blocks)


def test_default_preserves_dispatch_counts():
    assert CoreConfig().router_aux_count_source == "dispatch"


@pytest.mark.parametrize("source", ["fresh", True, None, 1])
def test_invalid_count_source_is_rejected(source):
    with pytest.raises(InputError, match="router_aux_count_source"):
        CoreConfig(router_aux_count_source=source)


def test_current_counts_fail_on_old_core_instead_of_silently_using_dispatch():
    config = SimpleNamespace(block=SimpleNamespace(routed_experts_router=SimpleNamespace()), block_overrides={})
    with pytest.raises(ValueError, match="updated Core runtime"):
        moe_models.prepare_model_config(
            config, SimpleNamespace(layer_types=[]), CoreConfig(router_aux_count_source="current")
        )
