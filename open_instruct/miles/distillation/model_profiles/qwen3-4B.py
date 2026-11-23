"""Megatron architecture profile for Qwen/Qwen3-4B-Base.

Same dimensions as the Miles copy (``scripts/models/qwen3-4B.py``: 36 layers, hidden 2560, FFN
9728, 32 heads in 8 query groups, head dim 128, RoPE base 1e6, tied embeddings), plus the explicit
``--padded-vocab-size`` described in ``qwen3-1.7B.py``: without it Megatron pads the vocabulary
to 152064 under TP2 and the mbridge conversion scatter fails on the embedding.
"""


def model_args() -> str:
    return (
        "--swiglu "
        "--num-layers 36 "
        "--hidden-size 2560 "
        "--ffn-hidden-size 9728 "
        "--num-attention-heads 32 "
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
