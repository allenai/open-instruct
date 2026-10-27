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
regressions pass. GPU serving and end-to-end RL qualification remain pending.

See [the GPU launch batch](experiment.md#five-scale-sglang-midtrain-qualification-2026-10-27)
and [the implementation guide](docs/miles/hybrid-mainline.md).
