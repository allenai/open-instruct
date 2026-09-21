# Exploratory gradient noise scale of MILES/Core GRPO

September 21, 2026. Mixed math/instruction-following/code/general workload, measured
at the update-100 policy of the completed packing-off arm. Reviewed before publication.

**The measured raw gradients are noise-dominated across the tested batch sizes.
This motivates investigating larger optimizer batches, but does not establish a
critical batch size or a calibrated 95% lower bound.** Production uses 64 retained
prompt groups, each containing four responses.

The initial report claimed a critical batch size of at least 2,024 groups at 95%
confidence. That interpretation is withdrawn: it conflated a gradient-noise proxy
with useful batch size and used an uncalibrated null-based confidence approximation.
The measurements below are retained with their limitations.

## What was measured

The simple gradient noise scale from [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162)
is `B_simple = tr(Sigma) / |G|^2`. It compares covariance of raw per-example gradients
with their squared population mean. Its relationship to useful training batch size
depends on optimizer, curvature and data distribution. [Ai2's Olmo study](https://allenai.org/blog/critical-batch-size)
found that this proxy did not reliably match empirical critical batch size.

The sampling unit here is a complete prompt group. Group-relative advantages make
responses within a group dependent; independence between groups remains an assumption.
These measurements do not include Adam's preconditioning or moment history.

## Panel and diagnostics

The probe used 3,072 responses in 768 groups of four from rollouts 88–99 of
`mixed-packing-recovery-20260920/off-v2`, differentiated at its update-100 HF export.
[Probe](https://beaker.org/ex/01M31PB5N8HGR05K45S1PGXSKR): 3.3 hours on one B300.

It differentiates each response's importance-weighted score function and stores a
uniform coordinate sample within each parameter family. Group gradients are
advantage-weighted sums of response scores. The probe/manifest/results are retained
under ignored `runs/critical-batch-20260921/`; reusable arithmetic is in
`scripts/miles/noise_scale.py`.

| Check | Value | Interpretation |
|---|---|---|
| Mean absolute engine log-probability gap | 0.026 nats/token, roughly flat over rollouts | Reassuring, but cannot distinguish numerical mismatch from policy drift |
| Importance weights clipped | About 2 in 100,000 tokens | Clipping is rare; importance weighting can still affect gradients |
| Sketched/exact response squared-norm ratio | 0.975 pooled | Checks aggregate magnitudes, not cancellation in group inner products |
| Rejected/skipped responses | 0 of 3,072 | No rejection due to length or memory in this panel |

This is an HF score-function probe with the old-policy reference reset to the
checkpoint being differentiated, not a replay of the live Core optimizer path.
Nearby-policy sampling and route/numerical differences remain limitations.

## Batch-gradient curve

For iid groups, `E|mean gradient of B|^2 = |G|^2 + tr(Sigma)/B`. Multiplying by `B`
gives an intercept representing noise and a slope representing coherent signal.

| Batch (groups) | 1 | 8 | 32 | 128 | 256 | 384 |
|---|---|---|---|---|---|---|
| B × mean squared batch gradient | 2.219 | 2.226 | 2.194 | 2.209 | 2.283 | 2.363 |

These descriptive means average eight random partitions of the same panel. Batches
are disjoint within a partition, not across partitions. The original error bars
incorrectly treated all partition batches as independent. At `B=1`, the reported
SE of 0.055 becomes approximately 0.155 when treating the 768 groups as independent.
Corrected tooling returns no population standard error for reused partitions or a
single available batch; it reports a conventional sample SE only for one partition
with at least two disjoint batches. The original error bars must not be reused.

The estimated squared population gradient is `1.19e-4`, versus an approximate null
standard deviation of `5.94e-4`: the panel does not resolve the coherent signal.
The point ratio of 18,585 groups is consequently unstable and is not an operational
batch-size recommendation.

A null-Gaussian calculation produced a putative floor of 2,024 groups. It omits
nonzero-mean variance, numerator uncertainty, group dependence and sketch uncertainty.
Its coverage has not been calibrated. The estimated effective dimension of 47 and
the uncorrected ratio of 738 groups are also diagnostics, not certified bounds.
A sample mean's squared norm exceeds the population squared norm in expectation,
not for every realized panel. Historical function/field names such as `lower_bound`
and `noise_scale_floor` are retained for reproduction, with explicit caveats.

## Length and filtering observations

Across three mixed runs, the online filter discards 47–50% of sampled prompt groups;
about 84% of discarded groups have four zero rewards. These are group fractions,
not fractions of generated tokens or GPU time. [Filter measurement](https://beaker.org/ex/01M31NS44Q8DWEWS20XZA7T8GS).

The correlation between log response length and log exact score-gradient squared
norm is −0.977. The shortest tenth accounts for 49.9% of the sum of exact response
score squared norms, or 35.2% after weighting those diagonal terms by squared
centered advantages. Neither quantity is the fraction of grouped policy-gradient
variance, which also includes cross-response terms and mean subtraction. The earlier
52% figure must not be described as policy-gradient variance.

Reweighting the sketches to token-mean reduction changes the point noise-scale
estimate from 18,585 to 4,944 groups, but neither weighting resolves a signal. It
is not evidence that changing loss normalization improves learning.

## Systems implications

A larger optimizer batch on fixed hardware can amortize a fixed planning/search
budget over more serial microbatches. Histogram construction and scoring still grow
with input size. Adding GPUs can instead keep trainer time per update roughly
constant, so the amortization is not automatic.

The recovery runs waited a median of about 455 seconds (packing off) and 469 seconds
(EP8 reference) for a batch, much longer than warmed trainer work. [Timing measurement](https://beaker.org/ex/01M32CPB1GQVAHP4DB3B8YY181).
Generation dominates this configuration; a larger batch does not itself accelerate
sampling. The noise-scale proxy does not justify the earlier claims of maximum sample
efficiency, mandatory learning-rate scaling, exact exchangeability of batches and
updates, or a 3.7-fold reduction in updates to a given result.

The operational question requires a batch/learning-rate comparison at matched data
budget, recording quality, policy age and separated generation/planning/training time.
No new training was launched for this review.

## Group-size extrapolation and remaining limits

An exploratory binary beta-binomial fit to estimated drop rates predicts fewer
all-wrong groups at larger fanout, but more generated responses per retained group.
The model undercounts total drops because rewards can agree at intermediate values.
These extrapolations were not validated by larger-fanout training. They should not
be treated as an established compute/learning tradeoff. Subsets of two and three
responses in this retained panel also fail to resolve a coherent gradient signal.

Captures are written after filtering, so the sampled-prompt population is not directly
measured. The panel contains no task labels; attributing its variance to disagreement
between math, instruction-following, code and general tasks remains a hypothesis.
