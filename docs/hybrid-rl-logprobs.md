# Hybrid Model RL: Recurrent State Leak and cu_seqlens Fix

## Problem

When running GRPO with hybrid (recurrent + attention) models like `allenai/Olmo-Hybrid-Instruct-DPO-7B`,
sequence packing causes **recurrent state to leak across packed sequence boundaries**.

In GRPO, multiple sequences are packed into a single tensor for efficient training. The attention mask
uses unique sequence IDs to indicate boundaries (e.g., `[1,1,1,2,2,0,0]`). Attention layers handle
this correctly via the mask, but the recurrent layers (`OlmoHybridGatedDeltaNet`) in HuggingFace
transformers do **not** — they treat the entire packed input as one contiguous sequence, carrying
SSM state from sequence 1 into sequence 2.

This manifests as a massive gap between vLLM logprobs (which processes sequences individually with
fresh state) and local HF logprobs (which processes packed sequences with leaked state).

## Impact: debug metrics comparison

### Without any fix (Mar 12, experiment `01KKHDH57HWM89ASVVPNB427BQ`)

Hybrid model, no packing fix. The recurrent state leak makes local logprobs nearly meaningless:

| Step | diff_mean | diff_max | diff_std | reverse_kl |
|------|-----------|----------|----------|------------|
| 1    | 15.87     | 41.97    | 8.57     | 11.43      |
| 2    | 11.34     | 38.43    | 7.11     | 8.10       |
| 3    | 11.29     | 41.15    | 9.21     | 8.63       |

### With cu_seqlens monkey-patch (Mar 16, experiment `01KKVTQQZ86A5PB1MV1C2337DQ`)

Hybrid model, with the cu_seqlens fix that resets recurrent state at sequence boundaries:

| Step | diff_mean | diff_max | diff_std | reverse_kl |
|------|-----------|----------|----------|------------|
| 1    | 0.46      | 18.28    | 1.31     | 0.24       |
| 2    | 0.58      | 18.47    | 1.45     | 0.28       |
| 3    | 0.24      | 18.15    | 0.81     | 0.10       |
| 4    | 0.41      | —        | 1.19     | 0.20       |
| 5    | 0.36      | —        | 1.02     | 0.13       |

### Non-hybrid baseline (Mar 6, experiment `01KK201Y3C2Z6VNJVKRPASEGHA`)

Standard transformer (Olmo-3-1025-7B), no recurrent layers, no state leak possible:

| Step | diff_mean | diff_max | diff_std | reverse_kl |
|------|-----------|----------|----------|------------|
| 1    | 0.11      | 13.19    | 0.50     | 0.04       |
| 2    | 0.11      | 13.05    | 0.47     | 0.04       |
| 3    | 0.06      | 11.26    | 0.33     | 0.02       |
| 4    | 0.10      | 11.52    | 0.42     | 0.03       |
| 5    | 0.09      | 10.65    | 0.37     | 0.03       |

### Summary

| Configuration              | diff_mean | reverse_kl | Status        |
|----------------------------|-----------|------------|---------------|
| Hybrid, no fix             | 11–16     | 8–11       | Broken        |
| Hybrid, cu_seqlens patch   | 0.24–0.58 | 0.08–0.28  | Healthy       |
| Non-hybrid baseline        | 0.06–0.11 | 0.02–0.04  | Healthy       |

The cu_seqlens patch brings hybrid logprob agreement to within ~4x of the non-hybrid baseline.
The remaining gap is expected — hybrid models use different numerical paths in vLLM vs HF
(triton kernels vs torch), so some difference is inherent.

## Root cause

The HuggingFace transformers `OlmoHybridGatedDeltaNet.forward()` does not pass `cu_seqlens`
to the FLA kernels (`chunk_gated_delta_rule`, `ShortConvolution`). Without `cu_seqlens`, these
kernels treat the entire input as one contiguous sequence.

The olmo-core implementation (`GatedDeltaNet.forward()`) handles this correctly by:

1. Deriving `cu_seqlens` from the attention mask via `get_unpad_data()`
2. Unpadding the hidden states with `index_first_axis()`
3. Passing `cu_seqlens` to both conv layers and the recurrent kernel
4. Re-padding output with `pad_input()`

## Fix: monkey-patch in grpo_fast.py

We monkey-patch `OlmoHybridGatedDeltaNet.forward` at worker initialization time
(`PolicyTrainerRayProcess.__init__`) to match the olmo-core approach. The patch:

1. Derives `cu_seqlens` from the packed attention mask (sequence ID boundaries)
2. Unpads hidden states to remove padding tokens
3. Passes `cu_seqlens` to `ShortConvolution` (q/k/v conv layers) and `chunk_gated_delta_rule`
4. Re-pads output to original shape

See `_patch_gated_deltanet_cu_seqlens()` in `open_instruct/grpo_fast.py`.

### Previous approach (removed)

Before the cu_seqlens patch, we used `_forward_packed_sequences_separately()` in `grpo_utils.py`,
which ran a separate `model()` forward pass for each packed sequence. This was correct but
**catastrophically slow with DeepSpeed ZeRO-3** because each forward pass triggers a full
NCCL all-gather of all model parameters. With N packed sequences, this multiplied the
communication overhead by N, causing training to hang indefinitely on multi-node setups.

## Upstream fix

The proper fix is for HuggingFace transformers to pass `cu_seqlens` in `OlmoHybridGatedDeltaNet.forward()`,
matching the olmo-core implementation. Until that happens, the monkey-patch is necessary for
any packed-sequence training with hybrid models.
