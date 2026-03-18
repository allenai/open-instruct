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

## Isolated logprobs comparison tests

The test in `scripts/test_logprobs_comparison.py` isolates the vLLM-vs-HF logprob gap
outside of training, using single (unpacked) sequences with the same datasets and
response lengths as production (8192 tokens, 6 datasets from the production mix).

Run on Beaker:
```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/run_logprobs_test.sh
```

### Production experiment config notes

The production experiments used different chat templates:
- **Hybrid** (`01KKVTQQZ86A5PB1MV1C2337DQ`): `--chat_template_name olmo123` (falls through to tokenizer's built-in `chat_template.jinja`)
- **Non-hybrid** (`01KK201Y3C2Z6VNJVKRPASEGHA`): `--chat_template_name olmo` (from `CHAT_TEMPLATES` dict in `dataset_transformation.py`)

The test replicates this: the hybrid model uses its built-in template; the transformer
model gets the `olmo` template from `CHAT_TEMPLATES` (since `Olmo-3-1025-7B` has no
built-in chat template).

### Results (Mar 18, experiment `01KM0QY7F73Y1JG6P7S7Q4PYYA`)

#### TestGRPOLogprobsMatch — vLLM generate, HF score (8192 response tokens, 6 prompts)

| Model | mean | max | std | median |
|-------|------|-----|-----|--------|
| Hybrid (Olmo-Hybrid-Instruct-DPO-7B) | 0.027 | 4.175 | 0.085 | 0.002 |
| Transformer (Olmo-3-1025-7B) | 2.394 | 17.333 | 2.358 | 1.743 |

These results are inverted compared to production: in the test the hybrid is better,
in production the hybrid is worse. See "Open questions" below for hypotheses.

#### TestVllmVsPackedHF — production-matching packed vLLM-HF comparison

*Results pending* — run with:
```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/run_logprobs_test.sh
```

This test directly replicates the production metric by comparing vLLM logprobs
against packed-HF logprobs scored with `forward_for_logprobs`. Two variants:
- **packed**: 2 sequences packed into 11264-token tensor
- **single**: 1 sequence in its own tensor (control)

## Open questions: why do test and production results diverge?

The test and production results show an inverted pattern:

|                  | Test (single seq, 8192 forced) | Production (packed, 8192 max) |
|------------------|-------------------------------:|------------------------------:|
| Hybrid           | 0.027                          | 0.24–0.58                     |
| Transformer      | 2.394                          | 0.06–0.11                     |

The hybrid gets *worse* going from test → production. The transformer gets *better*.

### Key difference: production uses packed scoring

Production compares vLLM logprobs against **packed** HF logprobs (via
`grpo_utils.forward_for_logprobs` with sequence-ID attention masks). Our baseline
`TestGRPOLogprobsMatch` uses `_hf_score` which `.clamp(0, 1)` on the attention mask,
losing sequence ID info. This means the baseline test doesn't replicate the exact
production scoring path.

Production also splits the metric by packing status:
- `debug/vllm_diff_mean_packed`: diff for tensors with >1 packed sequence
- `debug/vllm_diff_mean_unpacked`: diff for tensors with a single sequence

The 0.24–0.58 numbers may be from the packed metric only. If the unpacked metric
is ~0.03 (matching our test), then packing IS the explanation.

### TestVllmVsPackedHF: reproducing the production comparison

`TestVllmVsPackedHF` directly replicates the production metric:
1. Generate responses with vLLM (get vLLM logprobs)
2. Pack into tensors using `rl_utils.pack_sequences`
3. Score with `grpo_utils.forward_for_logprobs` (matching production exactly)
4. Compare vLLM logprobs vs packed-HF logprobs per response token

Two variants:
- **packed** (2 sequences in PROD_PACK_LENGTH=11264): matches production packed case
- **single** (1 sequence, tight pack): isolates `forward_for_logprobs` effect

If `packed` gives ~0.2–0.5 and `single` gives ~0.03, packing is the cause.

### Hypothesis 2: production responses are shorter than 8192 (transformer only)

Production uses `--stop_strings "</answer>"`, so many responses terminate well before
the 8192-token maximum. The transformer's sliding window gap scales steeply with length.
`TestNaturalResponseLength` tests this by generating with natural stopping.

## Upstream fix

The proper fix is for HuggingFace transformers to pass `cu_seqlens` in `OlmoHybridGatedDeltaNet.forward()`,
matching the olmo-core implementation. Until that happens, the monkey-patch is necessary for
any packed-sequence training with hybrid models.
