# H235 half-group gradient noise

This diagnostic uses the validated H235 V3 image `01M7FQHX6QCVFBEHT784XQTXJK`
as its base. The image overlay adds only these measurement files and replaces the
single native backward call with the guarded measurement wrapper. It does not
change the checked-in MILES runtime lock or any existing experiment.

The two committed configurations under
`configs/miles/experiments/h235-gradient-noise-20261116/` use H196 SFT and the
completed B512-Noff update-400 HF evaluation snapshot. Each collects eight fresh
512-response batches under the original Dolci priority recipe, with group size 8,
normalization off, response-mean loss, 32K responses, and unchanged admission/lag.
Learning rate is zero: policy weights stay fixed while native optimizer clocks and
publication continue. No evaluation or weight/checkpoint output is requested.
These are new diagnostic draws, not reconstructed historical batches.

For each batch, gather sample identities across trainer ranks, verify complete
sibling groups, and randomly select exactly half the groups. One additional full
forward/backward masks the other groups' advantages and doubles the objective.
Its gradient is the selected-half mean A. Snapshot optimizer-owned slices to CPU,
clear gradients, restore advantages/RNG, and run the normal full-batch backward F.
The other-half mean is B=2F-A. Thus:

- signal = A dot B = 2 A dot F - ||A||^2;
- full-batch noise = ||A-F||^2;
- noise scale in groups = groups * mean(noise) / mean(signal).

CPU snapshots cover the optimizer's named parameters, with EP-MP scaling applied
only to diagnostic copies. Replicated optimizer coordinates count once; sharded
coordinates sum across ranks. FP8 stores or unfamiliar ownership layouts fail
rather than silently omitting parameters. Compare the reported element count
across batches. Router/entropy/KL auxiliary objectives and token-mean loss are
explicitly unsupported. The ordinary optimizer receives the unmodified full-batch
gradient. One additional full-batch backward plus CPU transfers/reductions is
required per measurement; timing is reported separately. This intentionally
spends more than a sparse online probe to get a small isolated pilot first.

Group identity relies on native sample indices: eight consecutive indices form a
sibling group. Complete identity coverage and no duplicate indices are required.
The loss mask and per-response denominator are preserved when advantages are
masked, including any ungraded responses. Current/behavior probabilities, routing
replay, and importance weights remain native. All-zero groups remain filtered, so
the estimator describes the accepted training population, not unfiltered rollouts.

Use `summarize_gradient_noise.py` on each run's `gradient-noise.jsonl`. It averages
powers before dividing, retains negative signal measurements, and bootstraps
batches for a preliminary interval. Eight measurements and asynchronous sampling
limit precision. This is a raw optimizer-coordinate noise-scale proxy, before
clipping/Adam preconditioning, not proof of critical batch size or a schedule.
Source heterogeneity, selection, and correlations between groups remain relevant.

Build the committed overlay with:

```bash
beaker image pull 01M7FQHX6QCVFBEHT784XQTXJK h235-noise-base
docker build -f scripts/miles/diagnostics/Dockerfile.gradient-noise \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t h235-gradient-noise .
```

Publish that image, then use its immutable ID with the normal committed-image
wrapper for each config. Each run requests eight B300 GPUs on Holmes, normal
priority, 45 minutes allocated protection, 90-minute timeout and no automatic
retry; this is a bounded diagnostic, not an eight-hour training allocation.
Inspect scheduler events if queued. Record the image, experiment IDs, sample
coverage, numerical/finite checks, and timing before interpreting the estimates.
