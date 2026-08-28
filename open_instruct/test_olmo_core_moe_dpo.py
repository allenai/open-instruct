"""Tests that the MoE and dense DPO train modules share one objective.

The MoE train module exists because ``OLMoDDPModel`` refuses FSDP2, not because its DPO
objective differs. If the loss or the metric reduction is ever copied into one of them,
the two silently diverge and the difference reads as an architecture effect rather than
as a bug -- exactly the kind of result this arm exists to measure. These tests fail if
either class starts owning objective code.
"""

import pytest

from open_instruct import olmo_core_train_modules

# Every method that defines "what DPO computes", as opposed to "how a batch is stepped".
OBJECTIVE_METHODS = (
    "_init_dpo_objective",
    "_compute_microbatch_loss",
    "_accumulate_dpo_microbatches",
    "_record_dpo_metrics",
    "global_num_flops_in_batch",
)


def _moe_module():
    """The MoE train module, or skip: it needs an olmo-core revision with OLMoDDPTrainModule."""
    olmo_core_moe_dpo = pytest.importorskip(
        "open_instruct.olmo_core_moe_dpo", reason="olmo-core revision without OLMoDDPTrainModule"
    )
    return olmo_core_moe_dpo.MoEDDPDPOTrainModule


@pytest.mark.parametrize("method", OBJECTIVE_METHODS)
def test_mixin_owns_the_objective(method):
    assert method in olmo_core_train_modules.DPOObjectiveMixin.__dict__, (
        f"{method} should live on DPOObjectiveMixin so both train modules share one implementation"
    )


@pytest.mark.parametrize("method", OBJECTIVE_METHODS)
def test_dense_module_does_not_redefine_the_objective(method):
    assert method not in olmo_core_train_modules.DPOTrainModule.__dict__, (
        f"DPOTrainModule redefines {method}; it should inherit it from DPOObjectiveMixin"
    )


@pytest.mark.parametrize("method", OBJECTIVE_METHODS)
def test_moe_module_does_not_redefine_the_objective(method):
    assert method not in _moe_module().__dict__, (
        f"MoEDDPDPOTrainModule redefines {method}; it should inherit it from DPOObjectiveMixin"
    )


def test_both_modules_resolve_the_objective_to_the_same_functions():
    dense = olmo_core_train_modules.DPOTrainModule
    moe = _moe_module()
    for method in OBJECTIVE_METHODS:
        assert getattr(dense, method) is getattr(moe, method), (
            f"{method} resolves to different functions on the dense and MoE train modules"
        )


def test_moe_module_supplies_its_own_stepping():
    """The one thing the MoE module *must* override: OLMoDDP steps model_parts, not model."""
    assert "train_batch" in _moe_module().__dict__
