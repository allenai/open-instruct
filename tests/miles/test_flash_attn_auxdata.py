"""FA4 image-patch regressions, including CPU-only SM100 kernel compilation.

Run in the MILES image. CUDA tests also exercise real argument marshalling;
SM100 compile tests need no GPU and do not establish B300 numerical correctness.
"""

import warnings
from contextlib import contextmanager

import cutlass
import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute import runtime
from flash_attn.cute import interface, utils
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from torch.nn import functional as F


@contextmanager
def reject_auxdata_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Argument aux_data .*cannot be converted", category=UserWarning)
        yield


def fake(dtype, shape):
    return runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )


def sm100_compile_options(*, device_tuned):
    """Target the visible Blackwell device, or cross-compile for sm_100a.

    FA4's SM100 kernels read the DSL's detected architecture in their constructors
    and tune for it (B300 is sm_103 and skips exponent emulation). Forcing sm_100a
    on such a device configures the kernel for one chip and compiles it for
    another, which libNVVM rejects. Production compiles never pass an explicit
    architecture; mirror that on Blackwell. Arch-independent kernels still
    cross-compile for sm_100a on other GPUs; device-tuned FA4 kernels skip there.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11):
        return "--enable-tvm-ffi"
    if device_tuned and torch.cuda.is_available():
        pytest.skip("FA4 SM100 compilation requires no visible GPU or an SM100/SM110 device")
    return "--gpu-arch=sm_100a --enable-tvm-ffi"


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_sm100_attention_compiles_without_auxdata_warning(backward, packed):
    options = sm100_compile_options(device_tuned=True)
    prefix = () if packed else (cute.sym_int(),)
    q = fake(cutlass.BFloat16, (*prefix, cute.sym_int(), 8, 128))
    k = fake(cutlass.BFloat16, (*prefix, cute.sym_int(), 4, 128))
    lse = fake(cutlass.Float32, (*prefix, 8, cute.sym_int()))
    seqlens = fake(cutlass.Int32, (cute.sym_int(),)) if packed else None
    if backward:
        dq = fake(cutlass.Float32, (*prefix, 8, cute.sym_int(divisibility=128)))
        dk = fake(cutlass.Float32, (*prefix, 4, cute.sym_int(divisibility=128)))
        kernel = FlashAttentionBackwardSm100(128, qhead_per_kvhead=2, is_causal=True)
        args = (q, k, k, q, lse, lse, dq, dk, dk)
    else:
        kernel = FlashAttentionForwardSm100(
            128, qhead_per_kvhead=2, is_causal=True, is_static_persistent=False, is_varlen_q=packed
        )
        args = (q, k, k, q, lse)
    with reject_auxdata_warning():
        cute.compile(
            kernel,
            *args,
            cutlass.Float32(128**-0.5),
            mCuSeqlensQ=seqlens,
            mCuSeqlensK=seqlens,
            aux_data=utils.AuxData(),
            stream=cuda.CUstream(0),
            options=options,
        )


@cute.kernel
def aux_kernel(output: cute.Tensor, aux_data: utils.AuxData):
    value = cutlass.Float32(0)
    if cutlass.const_expr(aux_data.tensors is not None):
        value += aux_data.tensors[0][0]
    if cutlass.const_expr(aux_data.scalars is not None):
        value += aux_data.scalars[0]
    output[0] = value


@cute.jit
def aux_launch(output: cute.Tensor, aux_data: utils.AuxData, stream: cuda.CUstream):
    aux_kernel(output, aux_data).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@pytest.mark.parametrize("tensors,scalars", [(False, False), (True, False), (False, True), (True, True)])
def test_auxdata_empty_and_nonempty_compile(tensors, scalars):
    output = fake(cutlass.Float32, (1,))
    aux = utils.AuxData([output] if tensors else None, (cutlass.Float32(2),) if scalars else None)
    with reject_auxdata_warning():
        cute.compile(aux_launch, output, aux, cuda.CUstream(0), options=sm100_compile_options(device_tuned=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auxdata_runtime_values_are_not_frozen_at_compile_time():
    output = torch.empty(1, device="cuda")
    tensor = torch.tensor([3.0], device="cuda")
    output_cute, tensor_cute = [runtime.from_dlpack(t) for t in (output, tensor)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    with reject_auxdata_warning():
        compiled = cute.compile(
            aux_launch,
            output_cute,
            utils.AuxData([tensor_cute], (cutlass.Float32(2),)),
            stream,
            options="--enable-tvm-ffi",
        )
        for tensor_value, scalar_value in [(3.0, 2.0), (11.0, 7.0)]:
            tensor.fill_(tensor_value)
            compiled(output, utils.AuxData([tensor], (scalar_value,)), stream)
            assert output.item() == tensor_value + scalar_value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backward", [False, True])
def test_packed_attention_matches_reference(backward):
    if backward and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FA4 backward requires compute capability 9.x or newer")
    torch.manual_seed(17)
    lengths = [32, 65, 129]
    values = [
        torch.randn(sum(lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=backward)
        for heads in (8, 4, 4)
    ]
    refs = [v.detach().float().requires_grad_(backward) for v in values]
    seqlens = torch.tensor([0, 32, 97, sum(lengths)], device="cuda", dtype=torch.int32)
    reference = torch.cat(
        [
            F.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.repeat_interleave(2, dim=1).transpose(0, 1),
                v.repeat_interleave(2, dim=1).transpose(0, 1),
                is_causal=True,
            ).transpose(0, 1)
            for q, k, v in zip(*(x.split(lengths) for x in refs), strict=True)
        ]
    )
    with reject_auxdata_warning():
        output = interface.flash_attn_varlen_func(
            *values,
            cu_seqlens_q=seqlens,
            cu_seqlens_k=seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            causal=True,
        )[0]
        torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.02)
        if backward:
            gradient = torch.randn_like(output)
            output.backward(gradient)
            reference.backward(gradient.float())
            for value, ref in zip(values, refs, strict=True):
                torch.testing.assert_close(value.grad.float(), ref.grad, atol=0.03, rtol=0.05)
