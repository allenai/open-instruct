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

## First pilot: immutable source overlay

The first pilot avoids a 27 GB local image download by using the same immutable
image with a content-addressed source overlay. Source dataset
`01M7HCPYZ0FM7V1HXQRXTN566H` contains the committed Python files from
`434d49bcc`; its manifest SHA256 is
`8d29a722fa8d01689802f7bc0ea4efd0f4c5f81b8b2d404a8a22b2c3266e3e1d`.
CPU qualification `01M7HCRN5TBEAGP2V5YCZ0V7YW` materialized the overlay in its
own WEKA directory, passed all eight tests in the pinned image, and verified
both configs and checkpoints. Its output includes the patched actor hash.

The committed configs pin the manifest digest and use `BASH_ENV` to verify and
install the overlay into each disposable diagnostic container before native
startup. Only these jobs receive that environment variable. Files are copied
into the container-local application tree and the guarded actor call is patched
there; the runtime image, lock, original application source and existing runs
remain untouched. The launch receipt records both immutable image and overlay
pins. The Dockerfile remains an alternative for packaging a dedicated image.

Pilot submissions (normal priority, 45m protection / 90m timeout):

1. SFT: https://beaker.org/ex/01M7HCT6CTE9KDK20V2T60FKNJ
2. Final B512: https://beaker.org/ex/01M7HCTKZ6AS868RTXGP7YWEGV

Both were queued at 05:23 UTC on November 16. Scheduler events report insufficient
node slots and higher-ranked pending work. No GPU measurements were available at
that check. The local collector records latest attempts and scheduler events,
fetches only `run/gradient-noise.jsonl` after success, checks all eight steps and
full-model parameter coverage, and writes per-stage `summary.json` files:

```bash
python -m scripts.miles.diagnostics.watch_gradient_noise \
  --sft 01M7HCT6CTE9KDK20V2T60FKNJ \
  --final 01M7HCTKZ6AS868RTXGP7YWEGV \
  --output runs/h235-gradient-noise-20261116
```

## Retry after constructor failure

Both original pilots started on November 16 and failed before measurements:
OLMoDDPOptimizer requires `lr > 0` during construction. The corrected actor calls
`build_frozen_module`, which temporarily supplies 1e-6 only while constructing the
module, then restores args.lr=0 before the actor initializes CoreLRScheduler.
No forward/backward or optimizer update occurs during that constructor window.
Regression tests cover restoration on success/failure and unchanged weights under
zero scheduled LR. The collector now retries transient Beaker API failures.

Corrected source revision: `03ea19bbe`; immutable dataset:
`01M7JJJVCJKF2629F418424XZH`; manifest SHA256:
`53ca37cbd6489a81a1aef99f32ffc624a8295c3bd319300cc98c527a649c251e`.
Native-image CPU qualification `01M7JJM37KC06CQQTWVAG1QHKA` passed all ten
probe tests and validated both zero-LR configs/checkpoints. Local tests: 136 pass.
The original failed attempts and roots are retained; retries use `sft-r2` and
`final512-r2` under the same diagnostic parent.

1. SFT retry: https://beaker.org/ex/01M7JJNWMKETJNWZ93BH5QPNHK
2. Final B512 retry: https://beaker.org/ex/01M7JJP9Q9C1R6FMXS69GFQP4F

Collector output: `runs/h235-gradient-noise-20261116-r2/`. Only these retry IDs
should be used when checking for new measurements; the initial IDs are failed.

## Urgent allocated replacements

At Robert's request, replace the 45m-protected normal-priority pilots with urgent
allocated jobs, each protected for its entire 2h timeout (`min_runtime=2h`,
`timeout=2h`). Graceful stop budget is 110m. Source overlay and estimator remain
unchanged. The SFT predecessor was scheduler-preempted after five measurements;
the final-checkpoint predecessor was explicitly stopped before these submissions.
Partial r2 measurements remain in their original roots and local partial JSONs.
No in-flight batch resume is claimed: these jobs each collect eight fresh batches
from the same frozen checkpoint under fresh `*-urgent-r3` output roots.

1. SFT: https://beaker.org/ex/01M7JQ7NCBAREFQ3NTQHA7PK1R
2. Final B512: https://beaker.org/ex/01M7JQ7PB67XTAY2GH8QE8T339

Collector output: `runs/h235-gradient-noise-20261116-r3/`. Local verification:
136 tests pass; diagnostic Ruff lint/format checks pass. The earlier collector
was stopped; the replacement follows these IDs only.
