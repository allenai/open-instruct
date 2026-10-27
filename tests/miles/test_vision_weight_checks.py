"""Publication diagnostics preserve frozen tensors and still detect their corruption."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from miles.ray.rollout import server_cell
from miles.utils import weight_check_utils
from sglang.srt.utils import weight_checker


@pytest.mark.parametrize("vision", [False, True])
def test_destructive_weight_check_and_republication(vision, monkeypatch):
    # Keep the native comparison math while avoiding its GPU transfer in this CPU fixture.
    monkeypatch.setattr(weight_checker.RawComparable, "iter_chunks", lambda self: iter([(self.tensor.flatten(), None)]))
    args = SimpleNamespace(
        custom_generate_function_path="open_instruct.miles.datasets.vision_rollout.generate" if vision else None
    )
    model = torch.nn.ModuleDict(
        {name: torch.nn.Linear(3, 2) for name in ("language_model", "vision_tower", "connector")}
    ).requires_grad_(False)
    checker = weight_checker.WeightChecker(get_model=lambda: model, ps=None)
    checker.handle("snapshot")
    # GPU snapshots already copy to CPU; clone for this CPU-only fixture.
    checker._snapshot_tensors = {name: value.clone() for name, value in checker._snapshot_tensors.items()}
    original = {name: value.clone() for name, value in model.state_dict().items()}

    async def check_weights(*, action, allow_quant_error, selector, skip_list):
        assert selector == "all"
        return checker.handle(action, allow_quant_error=allow_quant_error, skip_tensor_list=skip_list)

    cell = object.__new__(server_cell.ServerCell)
    cell.args = args
    cell._state = server_cell.StateDisposed()
    client = SimpleNamespace(check_weights=check_weights)
    monkeypatch.setattr(server_cell.ServerCell, "api_client", property(lambda self: client))
    asyncio.run(cell.check_weights("reset_tensors", False, "all", None))
    for name, value in model.state_dict().items():
        if vision and not name.startswith("language_model."):
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        else:
            assert not torch.equal(value, original[name])
    restored = {name: value for name, value in original.items() if not vision or name.startswith("language_model.")}
    model.load_state_dict(restored, strict=not vision)
    compare_skip = weight_check_utils.skip_list_for_action(args, "compare", None)
    assert compare_skip is None
    asyncio.run(cell.check_weights("compare", False, "all", compare_skip))
    model["connector"].weight.add_(1)
    with pytest.raises(Exception, match="connector.weight"):
        asyncio.run(cell.check_weights("compare", False, "all", compare_skip))


def test_reset_preserves_explicit_skips_without_mutating_them():
    args = SimpleNamespace(custom_generate_function_path="open_instruct.miles.datasets.vision_rollout.generate")
    explicit = ["custom_buffer"]
    assert weight_check_utils.skip_list_for_action(args, "reset_tensors", explicit) == [
        "custom_buffer",
        "vision_tower.",
        "connector.",
    ]
    assert explicit == ["custom_buffer"]
    assert weight_check_utils.skip_list_for_action(args, "checksum", explicit) is explicit
