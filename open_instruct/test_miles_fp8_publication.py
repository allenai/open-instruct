"""Refresh lifecycle tests; GPU qualification separately exercises native kernels."""

from types import SimpleNamespace

import pytest
import torch

from open_instruct.miles.execution import fp8_publication as publication


def _method_type(*, transpose=False, serialized=False, block=False):
    class Method:
        quant_config = SimpleNamespace(get_name=lambda: "fp8", is_checkpoint_fp8_serialized=serialized)
        block_quant = block

        def create_weights(self, layer, shape, *, dtype, weight_loader):
            layer.weight = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.weight.weight_loader = weight_loader
            layer.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        def restore_weights_before_loading(self, layer):
            return "native restore"

        def process_weights_after_loading(self, layer):
            scale = layer.weight.float().abs().max().clamp_min(1e-10) / 448
            weight = (layer.weight.float() / scale).to(torch.float8_e4m3fn)
            layer.weight = torch.nn.Parameter(weight.t() if transpose else weight, requires_grad=False)
            layer.scale = torch.nn.Parameter(scale.reshape(1), requires_grad=False)

    return Method


@pytest.mark.parametrize("transpose", [False, True])
def test_repeated_refresh_preserves_values_scales_addresses_and_loader(transpose):
    method_type = _method_type(transpose=transpose)
    publication._install_method(method_type, lambda _: True)
    wrapped = method_type.create_weights
    publication._install_method(method_type, lambda _: True)
    assert method_type.create_weights is wrapped
    method = method_type()
    layer = torch.nn.Module()
    loader = object()
    initial = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    method.create_weights(layer, initial.shape, dtype=initial.dtype, weight_loader=loader)
    layer.weight.data.copy_(initial)
    method.process_weights_after_loading(layer)
    captured = dict(layer.named_parameters())
    pointers = {name: param.data_ptr() for name, param in captured.items()}
    baseline = layer.weight.detach().float().clone()
    for source in (initial, initial.flip(0) * 3, initial):
        method.restore_weights_before_loading(layer)
        assert layer.weight.dtype == torch.bfloat16
        assert layer.weight.shape == initial.shape
        assert layer.weight.weight_loader is loader
        # Separate buckets must survive until end_weight_update, not be
        # quantized independently or overwrite one another.
        layer.weight.data[0].copy_(source[0])
        layer.weight.data[1].copy_(source[1])
        method.process_weights_after_loading(layer)
        scale = source.float().abs().max() / 448
        expected = (source.float() / scale).to(torch.float8_e4m3fn)
        torch.testing.assert_close(layer.weight.float(), (expected.t() if transpose else expected).float())
        torch.testing.assert_close(layer.scale, scale.reshape(1))
        for name, param in layer.named_parameters():
            assert param is captured[name]
            assert param.data_ptr() == pointers[name]
    torch.testing.assert_close(layer.weight.float(), baseline)
    assert not hasattr(layer, "_oi_fp8_serving_parameters")


@pytest.mark.parametrize("serialized,block", [(True, False), (False, True)])
def test_other_fp8_modes_keep_native_restore(serialized, block):
    method_type = _method_type(serialized=serialized, block=block)
    publication._install_method(method_type, lambda _: False)
    layer = torch.nn.Module()
    method = method_type()
    method.create_weights(layer, (2, 3), dtype=torch.bfloat16, weight_loader=None)
    assert not hasattr(layer, "_oi_fp8_allocation")
    assert method.restore_weights_before_loading(layer) == "native restore"


def test_unsupported_backend_fails_before_mutating_parameters():
    method_type = _method_type()
    publication._install_method(method_type, lambda _: False)
    layer = torch.nn.Module()
    method = method_type()
    method.create_weights(layer, (2, 3), dtype=torch.bfloat16, weight_loader=None)
    original = layer.weight
    with pytest.raises(RuntimeError, match="requires CUDA"):
        method.restore_weights_before_loading(layer)
    assert layer.weight is original


def test_nested_refresh_rejected_without_discarding_loading_buffers():
    method_type = _method_type()
    publication._install_method(method_type, lambda _: True)
    layer = torch.nn.Module()
    method = method_type()
    method.create_weights(layer, (2, 3), dtype=torch.bfloat16, weight_loader=None)
    layer.weight.data.fill_(1)
    method.process_weights_after_loading(layer)
    method.restore_weights_before_loading(layer)
    loading = layer.weight
    with pytest.raises(RuntimeError, match="already open"):
        method.restore_weights_before_loading(layer)
    assert layer.weight is loading


def test_layout_change_rejected_before_copying_any_serving_parameters():
    method_type = _method_type()
    publication._install_method(method_type, lambda _: True)
    layer = torch.nn.Module()
    method = method_type()
    method.create_weights(layer, (2, 3), dtype=torch.bfloat16, weight_loader=None)
    layer.weight.data.fill_(1)
    method.process_weights_after_loading(layer)
    captured = layer.weight
    before = captured.float().clone()
    method.restore_weights_before_loading(layer)
    layer.weight = torch.nn.Parameter(torch.full((3, 3), 2.0), requires_grad=False)
    with pytest.raises(RuntimeError, match="serving layout"):
        method.process_weights_after_loading(layer)
    torch.testing.assert_close(captured.float(), before)


@pytest.mark.parametrize("precision", ["fp8", "nvfp4", None])
def test_olmo_cached_load_targets_refreshed_only_for_fp8(precision):
    sentinel = object()
    weights = iter([("weight", sentinel)])

    class Model:
        quant_config = SimpleNamespace(get_name=lambda: precision) if precision else None
        _weight_targets = sentinel

        def load_weights(self, actual, *, option):
            assert actual is weights
            assert option is sentinel
            assert self._weight_targets is (None if precision == "fp8" else sentinel)
            return sentinel

    publication._install_loader(Model)
    wrapped = Model.load_weights
    publication._install_loader(Model)
    assert Model.load_weights is wrapped
    assert Model().load_weights(weights, option=sentinel) is sentinel
