# Full-tag forced-exit stopping pilot

September 26, 2026. The user authorized a machinery check; learning runs still
require review. Source is isolated on `robertb/miles-forced-exit-machinery` so
concurrent group-size and other planning changes are not included.

## Experiment

Keep the same **4T non-EMO SFT step5402** checkpoint. Its `</think>` delimiter is
multi-token. H025 has single-token tags but a different, 2T pretraining lineage;
we do not substitute it or edit the selected checkpoint's tokenizer.

Each prompt produces four independent natural responses. Only those responses
enter GRPO/TIS and prompt-group reward normalization. Uniformly select one parent
per prompt, then up to five paragraph boundaries spread through its thinking.
Each cut supplies three forced answers. All trials are labels only, not trainer
samples. Boundaries reserve 1,024 answer tokens within the response limit; short
traces have fewer cuts. Natural closing-tag detection handles BPE merges with
punctuation/newlines. No duplicated delimiter is appended after a natural close.

For each parent, compare each cut's mean correctness against the mean over its
natural reward and all forced trials. No position decay is used initially.
All-zero and all-correct cut groups have zero auxiliary advantage: this pilot
cannot establish a stopping benefit on the entire all-correct wasted-tail
population. The contrast is observational, not an independently sampled estimate
of continuation value. A failed late cut can have negative advantage due to an
earlier successful cut; do not interpret every negative as causal evidence that
ordinary continuation from that state helps.

Every truncated response receives reward **zero**, even if a correct answer
appears in its unfinished text. Natural truncations retain their loss masks and
participate in group centering. Finished responses are graded only after the last
`</think>`; absent/empty final sections also score zero. These rules apply to both
arms and evaluation. Zero-variance filtering is disabled in both arms so auxiliary
outcomes never affect prompt admission. Record natural-signal, stopping-only and
no-signal group counts.

## Full-tag auxiliary training

Each cut has one teacher-forced prefix context ending in the full delimiter,
shared by its answer trials. Only the delimiter's sequence log probability enters
the auxiliary loss. This requires one scoring forward plus one training forward
per cut, including backward through its prefix; it is not the cheap single-token
parent-logit shortcut. Answer suffixes are never trained or forwarded by the
trainer. Prefix/forced behavior probabilities from inference are never used as
fabricated model scores.

Cuts with exactly zero advantage skip auxiliary scoring/training. Their labels
and natural-state captures remain available, and averaging still divides by the
original number of cuts. This removes zero-gradient work without reweighting the
remaining guidance. Rank padding is computed after this omission.

The anchor is a fresh detached actor score before the optimizer update. The loss
is a clipped sequence-ratio surrogate with coefficient **0.1**, clipping **0.2
below / 0.28 above**. Average over cuts, weight each sampled parent by the inverse
sampling fraction, then normalize by the natural batch size. The objective is
explicit forced-action guidance, not an unbiased importance-corrected gradient.
No classification head, answer-agreement label, or length reward is added.

Natural forwards retain serving-route replay. Auxiliary contexts route under the
current trainer; both anchor and training passes use that routing policy. Router
auxiliary coefficients must be zero. All trainer ranks execute an equal number
of auxiliary forwards/backwards; ranks with fewer cuts use zero-loss placeholders.
Natural and auxiliary microbatches share one Core gradient-accumulation call and
one optimizer step. Auxiliary samples have zero native metric count, preserving
the GRPO denominator. Publication is synchronous barrier, policy lag zero.

## Runs

Ignored configs/artifacts: `runs/forced-exit-math-20260926/` in the isolated checkout.
Maintained examples are unchanged. All jobs use one Beaker task/one replica on
`ai2/holmes`, workspace `ai2/open-instruct-dev`, budget `ai2/oe-other`.

| Run | GPUs | Updates | Scope |
|---|---|---:|---|
| `qualification.json` | EP2 trainer + two TP1 engines = 4 | 4 | Authorized actual-model machinery check, 32 natural responses/update |
| `baseline-clean-unallocated.json` | EP2 trainer + six TP1 engines = 8 | 128 | Authorized after qualification; ordinary natural GRPO, 64 responses/update |
| `forced-clean-unallocated.json` | EP2 trainer + six TP1 engines = 8 | 128 | Authorized after qualification; matched GRPO plus full-tag guidance |

Qualification protects one hour and has a two-hour hard timeout, with an explicit
one-hour driver budget. No checkpoint saving/HF export; whole-group rollout capture
and readiness capture each update. Learning arms must complete their update budget;
they have no four-hour soft cutoff. Review their hard timeout after measuring
qualification throughput. Their comparison needs both matched-update and measured
compute reporting; forced generation and auxiliary prefix training add work.

The active qualification replacement uses low priority and `minRuntime=0s`, as
requested. Cancel the original allocated backup only after the replacement
finishes. Learning runs also use unallocated scheduling with automatic resume.

Math mix: 8,192 pinned Open Reasoner Math + 2,048 GSM8K training examples (80:20),
384 Math + 128 GSM8K held-out examples. Qualification uses 128 + 32 training and
8 + 8 evaluation examples. Both learning arms share seed 17, LR 1e-6, 8,192 response
and 10,240 context tokens, BF16 head, packing/recomputation, offline W&B, zero router
auxiliary losses, and identical reward/filter rules. The learning arms use
`techarb/gsm8k-cleaner` revision `3a4e9e3e600ea2854a6d2d0483721b3cd68580ce`.
Save native recovery checkpoints every eight updates, retaining the latest two
completed checkpoints; export final HF weights for later frozen audits.

## Evidence and qualification gates

Inspect finite gradients and parameter changes, intact natural behavior scores,
full-delimiter score/training agreement, both auxiliary advantage signs, cut
coverage, one publication version, and captured finite hidden states. A machinery
pass is not a learning claim. If forced-answer truncation exceeds roughly **2%**
over the qualification pool, review the answer budget before approving learning;
never relax the zero-truncation reward rule to pass this gate.

Record full-tag probability separately at positive/negative-advantage cuts, request
latencies (including queue time), auxiliary scoring time, contexts/tokens, update
wall time and serving/cache metrics. Request latency sums are not engine compute
time. Clipping is not a hard bound on parameter updates; measure gradient scale.

Capture natural-parent LM-head inputs before each cut, parent correctness/final
answer, all trial rewards/answers, truncation, absolute/relative cut positions and
policy step. Validate packed offsets and CP=1. These are readout data, not a fitted
mechanistic probe. Use prompt-held-out splits and a position-only baseline later.
Report correct-exit/wrong-parent recovery separately from wrong-exit/correct-parent
premature stopping. Parse/normalize answers before comparing; agreement and entropy
are diagnostics only. Three trials do not establish readiness or precise entropy.

Retain unforced held-out accuracy, response lengths, truncation and paired answer
changes by task. A shorter correct trajectory is evidence for better stopping;
improved math accuracy is a separate result. Historical ready-then-lost and wasted-
tail estimates require independent confirmation of noisy first crossings.

CPU tests cover full-tag gradients, natural-only GRPO normalization, trial sharing,
BPE boundaries, reward gates, packing, two-rank auxiliary padding/reduction,
readiness offsets, cancellation and config guards. They do not qualify GPU kernels,
remote paths or throughput. The stale options snapshot's OLMo-core source identity
was repaired to match the existing runtime lock; native options did not change.
