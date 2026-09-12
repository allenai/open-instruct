# Opt-in runtime-row SwiGLU integration

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The active Core worktree is `robertb/miles-rl-adapter`, based on
`codex/small-hero-hf-20260909` at `b1fd2c9746e88baeb20e372bdca340d788d0f7e5`.
The earlier `robertb/miles-adapter` branch remains based on Jacob's
`jacobm/moe-v2-core-gdn2` at `169b8f9d06bce0276143876c82f630af483b03b7`.
The runtime lock records the exact promoted Core revision and source patch.

## Selection and isolation

`RoutedExpertsConfig.row_specialization` defaults to `static`. The field is
validated when constructing the routed-expert module, stored on that module,
and explicitly passed from its forward-only activation dispatch into
`swiglu_valid_prefix`. There is no environment switch or scoped mutable state.

The kernel wrapper also defaults to `static`, preserving callers that do not
pass a selector. `dynamic` selects a separate forward kernel whose row capacity
is a runtime argument with `do_not_specialize`, and whose row stride uses the
runtime launch-grid size. The original static forward kernel and the backward
kernel are unchanged. Arithmetic options, including `match_eager_rounding`,
remain independent and are passed identically to either forward kernel.

Direct wave-based EP callers continue to use the static wrapper default. Their
custom autograd does call these kernels during training; their isolation is
provided by the default argument, not by assuming all training avoids no-grad
execution. Backward capacity specialization remains a separate qualification item.

Open-instruct exposes `core.row_specialization`, also defaulting to `static`.
The three starting profiles and SFT GSM8K harness explicitly select `dynamic`.
The selected mode is copied into every routed-expert block configuration before
model construction, including layer overrides. The same construction applies to
a reference model. Dense standard Olmo 3 continues through its separate backend.

Rollback is `core.row_specialization="static"`. Checkpoint architecture comparison
ignores only this validated field within `routed_experts`, so older manifests
without the field and checkpoints from either mode remain resumable. New manifests
still record the full model configuration for provenance. Other model and topology
checks remain active.

## Acceptance evidence

The existing [EP2 successive-batch qualification](core-score-variants-20260911.md)
compared the forward runtime-row transformation on the full SFT model: 154,531 exact
response log-probabilities, one initial SwiGLU variant per rank and none on subsequent
batches. That isolated candidate is the basis of this dynamic kernel; it is not a
measurement of the new configuration selector or a new production training run.

The integration adds local GPU coverage of:

- static/dynamic exact output equality for BF16, FP16 and FP32, with both rounding
  modes, nonzero starts, empty valid ranges, partial column tiles and capacities
  on either side of the row-program cap;
- no additional dynamic-kernel compilation as capacities change;
- explicit dispatch selection and exact eager-path input gradients;
- paired full Core score/train/score cycles, including KDA and latent MoE, with
  activation recomputation on and off;
- exact model updates and native optimizer states, plus pre-clipping native
  gradient buffers, across modes;
- legacy checkpoint compatibility and rejection of unrelated architecture changes.

The warmed full-scorer repeats in the earlier EP2 experiment were approximately
1.2–1.3 seconds in both arms. They do not establish pretraining kernel performance.
A production timing follow-up should record per-rank SwiGLU compilation misses,
scoring wall time, token counts and cache state together. Token/time correlation
is a diagnostic, not a zero-correlation acceptance threshold: real compute can
scale with tokens after compilation stalls have been removed.

## Local integration result

Passed on the RTX 4090 using the existing compiled runtime with the current source
worktrees mounted read-only:

- 25 Core kernel/config/dispatch cases, including unchanged backward tests.
- Four paired KDA / latent-MoE score/train/score cases (recomputation on/off), with
  exact scores, pre-clipping native gradients, model updates and optimizer states.
  These are single-rank integration tests; EP2 evidence is the earlier isolated
  full-model experiment linked above.
- 42 runtime regression cases, including native checkpoint/resume, rollback/legacy
  config comparison, named-block initialization, parser contract and dispatch.
- 52 CPU interface tests; Core Black/isort/Ruff checks and open-instruct
  `make style` / `make quality` passed.

Core commit: `307d20590f241222176f51f1760529988402e1b7`. The locked Core source delta
was reconstructed against its pinned hero base and matched the recorded patch
checksum exactly. AST comparison confirmed the static forward and backward
functions are unchanged from Core290d2ca, and the new dynamic kernel matches the
qualified isolated candidate apart from its function name.

The paired integration test caught a missing traversal of Core's named block
configurations; initialization now handles a single block, named block dictionaries,
and per-layer overrides. Gradient comparison observes the optimizer's native buffers
before clipping rather than `.grad` tensors after the update.

A [warmed local screen](core-row-specialization-20260911-kernel-benchmark.json)
used BF16, hidden width 1024, eager rounding, 75% valid rows, and randomized mode
order. Median milliseconds per launch (30 samples, 20 launches per sample):

| Row capacity | Static | Dynamic |
| --- | ---: | ---: |
| 256 | 0.01080 | 0.01065 |
| 4096 | 0.01084 | 0.01070 |
| 16384 | 0.02611 | 0.02682 |
| 32768 | 0.16394 | 0.16415 |

This bounded screen ranged from about 1.4% faster to 2.7% slower for dynamic rows.
Small cases can be launch-limited. It does not establish a pretraining speedup or
regression, and no new full-checkpoint RL learning/timing run is claimed here.
Backward dynamic-row support remains unimplemented and unqualified.
