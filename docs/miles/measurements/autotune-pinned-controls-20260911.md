# Fixed FLA tuner choices eliminate the measured prefill divergence

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Two independent fresh-cache processes using the original Core image and matched serving settings had different prefill activations; five of seven observed FLA autotuner choices differed. Pinning all seven to the first reference process's declared B300 configurations made both new processes exactly match one another **and the original reference** on all four fixed prefixes.

The CPU comparison verified every captured activation value, router logit, selected expert ID (including slot order), expert weight, and full final-vocabulary logit. Each of three comparisons includes four cross-process cases and two within-process repeats; all 18 comparisons are exact. Control responses also agree. Both pinned processes exited 0, and the independent comparison exited 0. [Compact assertions and full-report SHA256s](autotune-pinned-controls-20260911.json).

Each armed request recorded 128 wrapped tuner calls, covering all seven pinned tuners with none missing. The two processes have identical recorded invocation sequences. Configurations were validated against declared hardware candidates before installing singleton choices, before warmup. Empty autotuner caches after singleton installation are expected; actual invocation records establish that the choices were used.

This identifies a cause of the **observed process-to-process fixed-prefix divergence**. It does not identify the responsible individual tuner, prove arbitrary shape or hardware equality, cover incremental decode or continuous batching, or establish how much of the original 100-update learning gap came from inference variability. Training and serving route agreement and auxiliary gradients remain separate questions. Active learning runs have not been changed.

Runs:

1. Pinned A: [Beaker](https://beaker.org/ex/01M27J3Y2J4HSG1N163EN1JJ83).
2. Pinned B: [Beaker](https://beaker.org/ex/01M27J4XKG5F72RW8ZXHCV54ZK).
3. Pinned twins and both reference comparisons: [Beaker](https://beaker.org/ex/01M27K4Z7111KFVAMVQA0KJGKV).

Image: `01M26N80T0V9PREQTS87J849P8`. Capture launch source: `f0a254294`; CPU comparison source: `255f3f4a806475c40e9cd1d135b5f9be2d1de2f2`. Pin profile: `configs/miles/diagnostics/autotune-reference-hfmatched-a.json`. Tensor captures remain under the original campaign's `update-zero-20260911-pina-{a,b}/core` directories; full comparison reports are also in Beaker results.
