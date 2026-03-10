# Stub module for flash_attn_2_cuda when CUDA kernels are not compiled.
# Installed as flash_attn_2_cuda so that flash_attn's triton-based ops
# (e.g. rotary embeddings used by vLLM) can be imported without the
# CUDA extension. Any actual call to CUDA flash attention will error.


def _not_compiled(*args, **kwargs):
    raise RuntimeError(
        "flash_attn_2_cuda was not compiled. "
        "Install flash-attn with CUDA support to use flash attention CUDA kernels."
    )


fwd = _not_compiled
bwd = _not_compiled
fwd_kvcache = _not_compiled
varlen_fwd = _not_compiled
varlen_bwd = _not_compiled
