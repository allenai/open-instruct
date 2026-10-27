# Research

## Temporary Olmo Hybrid MILES port

Status: ACTIVE.

Can the five released MainlineLadder sizes serve correctly under SGLang and then
support native Core RL without maintaining another inference fork?

Decision: reuse the release FLA Gated DeltaNet and keep its negative-eigenvalue
semantics; use SGLang RadixAttention for full attention. Restrict the temporary
bridge to eager TP1 with no prefix sharing or chunked prefill. GDN states are
per request slot and outside SGLang's allocator. This trades serving throughput
for a smaller implementation and a direct correctness check.

CPU finding: the actual pinned Core runtime builds the 275m native descriptor on
meta and matches all 192 released HF tensor names and shapes. Strict interchange
regressions pass. The corrected 275m GPU serving test passed (including reload, long prompts, and slot reuse); the expanded queued-request/FP32-state 275m check also passed. All five scales now pass with the BF16 reference-backbone fallback; end-to-end RL qualification remains pending.

Harness finding: the pinned SGLang returns a prompt-logprob suffix with an initial
`None` even when logprob_start_len is nonzero. Align by original token position
instead of dropping that entry and shifting the reference. This caused the first
batch attempt to fail independently of model correctness; CPU regressions cover it.

See [the GPU launch batch](experiment.md#five-scale-sglang-midtrain-qualification-2026-10-27)
and [the implementation guide](docs/miles/hybrid-mainline.md).


Serving finding (2026-10-27): all five final-midtrain sizes passed the BF16
reference-backbone fallback on Rhea: 160 functional requests and 40 dense probes,
including long prompts, queued requests, slot reuse, and disk reload. Maximum
selected-token logprob error stayed below 5e-4 and greedy agreement was 100%.
This selects `MILES_HYBRID_REFERENCE_BACKBONE=1` with BF16 logits for the temporary
recipe. It localizes the earlier discrepancy to differences introduced by the
faster packed/RadixAttention path, without isolating one specific operation.
That path remains unqualified across the whole family. The fallback is slower
(roughly 62–67 output tokens/s at 275m and 20–22 at 2.7b for these warm batches).
HF cached-versus-full discrepancies remain, so native RL publication, importance
ratios, restart recovery, and learning evidence are still pending; the overall
MILES integration question remains ACTIVE.
