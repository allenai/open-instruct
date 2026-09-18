"""Megatron architecture profile for Qwen/Qwen3-1.7B-Base (revision ea980cb0a6c2… as prepared).

Same dimensions as the Miles copy (``scripts/models/qwen3-1.7B.py``: 28 layers, hidden 2048, FFN
6144, 16 heads in 8 query groups, head dim 128, RoPE base 1e6, tied embeddings), plus an explicit
``--padded-vocab-size``. Megatron pads the 151936-token vocabulary to a multiple of 128 x TP
(152064 under TP2) unless told otherwise, and mbridge 0.15.1 scatters the unpadded HF embedding
across the tensor-parallel ranks as-is, so the conversion fails with a size mismatch. 151936 is
already a multiple of 128, so pinning it keeps the checkpoint and the HF export the same shape.
"""


def model_args() -> str:
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 2048 "
        "--ffn-hidden-size 6144 "
        "--num-attention-heads 16 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--rotary-base 1000000 "
        "--vocab-size 151936 "
        "--padded-vocab-size 151936 "
        "--kv-channels 128 "
        "--qk-layernorm "
    )
