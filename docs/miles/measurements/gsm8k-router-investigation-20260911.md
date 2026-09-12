# Router, response length, and serving reproducibility investigation

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The auxiliary-loss and serving/trainer routing hypotheses remain open. The new
observations establish a separate serving reproducibility issue before any trainer
initialization; they do not yet explain the final learning gap.

## Measured

- The [original100 length audit](response-lengths-100-20260911.md) verified all
  212 retained training/evaluation dumps against the prior hashes and token/reward
  proofs. Both trainers' responses shorten late. Training-wide mean length is1816
  tokens for Core and1898 for Megatron; in rollouts80–99 it is1512 versus1415.
  Megatron's late held-out improvement accompanies shortening, while Core's final
  capped tail grows. Among questions correct and uncapped at both updates60 and100,
  mean length falls1115→989 for Core (84 questions) and1240→688 for Megatron
  (85 questions). These are separate conditioned subsets, not identical cross-arm
  membership. [Paired IDs and conditional analysis](response-lengths-conditional-20260911.md)
  preserve that distinction. These are descriptive associations from one run per backend.
- The original shared128-question greedy evaluation scores are Core97→93 and
  Megatron96→107. Median length falls in both; it is not accurate to describe the
  entire Core distribution as moving uniformly toward longer responses.
- Four identical token prefixes from originally divergent generations were traced
  in the original Core and Megatron serving images, directly from the same shared
  HF file before trainer initialization. Their first captured differences occur
  in KDA attention outputs, before their first changed expert assignment. Python
  sources, six SGLang native shared libraries, dependency versions, and recorded
  numerical controls match. Resolved serving pool and sampler settings differ.
- Instrumented/uninstrumented requests and the repeated first case are exact
  within each original diagnostic process. These observations therefore distinguish
  process-to-process variability from repeatability within a process.
- A second independent launch of the original Megatron image/recipe changes the
  greedy next token on three of these four prefixes before trainer initialization:
  q341 changes5629→6914; q9752011→706; q1039358→6227. q605 remains13 but its log
  probabilities differ. The first activation difference is layer0 KDA attention in
  all four cases (maximum absolute error9.54e-7–1.91e-6); final-logit differences
  grow to0.50–0.796875. Expert sets first differ in layer1. This control does not
  switch trainer backend. The prefixes
  were selected to diagnose earlier divergence, so3/4 is not an estimate over
  ordinary prompts. [Retained replicate evidence](update-zero-megatron-replicates-20260911.json).
- Both initial trainer publications passed the full serving-weight comparison.
  In each backend on all four prefixes, captured activations, route logits/sets, and
  full next-token logits are exactly unchanged before versus after publication.
  Megatron v3 completed successfully. The Core job subsequently
  failed its180-second learner-disposal timeout; the successful observations are
  preserved, and the complete job/protocol is not reported as passing.

## What these observations do not establish

A large behavioral change can be learned while the difference between two learning
trajectories is still affected by initial numerical perturbations and sampling.
One seed per trainer does not establish a systematic backend effect. Neither
response-length association nor larger router parameter drift establishes which
objective caused an outcome.

The earlier1.132% figure is relative L2 error between gradient tensors (cosine
0.999937), not simply a1.132% difference in their overall magnitudes. Adam's
approximate invariance to uniform gradient scaling is not sufficient to
exclude optimizer effects: its epsilon, gradient mixtures, clipping, and state
history matter. Likewise, scoring-rounding error need not be zero-mean or merely
blur learning. Those mechanisms require measurements rather than a presumed sign.

The measured padded-token fraction in Megatron makes its auxiliary objective worth
examining. It does not prove all pad hidden states are identical at every layer,
or establish the direction or magnitude of the router's resulting update. Both
policy and auxiliary gradients must be compared on the same fixed samples.

A serving/trainer top-k disagreement is an observable forward discrepancy, not by
itself proof that an optimizer update is invalid. We need the corresponding log
probabilities, masks, replay mode, and policy/auxiliary gradient effects. Replay is
off in the original100 and active500 comparisons.

## Work in progress

1. Two independent original-Core-image, one-GPU HF controls use matched serving
   settings and fresh compiler caches. They record populated autotuner choices:
   [replicateA](https://beaker.org/ex/01M27GC5JWB8Z8TK9Z1WK5VFM8) and
   [replicateB](https://beaker.org/ex/01M27GCMH4D4T6X2C3X99XTFYA).
   They perform no trainer initialization or publication. Same selected cache
   entries alone will not prove every kernel launch identical.
2. Instrument actual Core and Megatron scorers on the same four prefixes and the
   retained first16-sample training batch, without optimizer steps. Capture token
   and layer alignment, expert sets and weights, log probabilities, and observer
   controls before interpreting serving/trainer route agreement.
3. Preserve completed intermediate Megatron500 checkpoints outside its live save
   tree so native rolling retention cannot remove the drift-analysis inputs.
   [CPU retention job](https://beaker.org/ex/01M27GYGMCX5SQYFZ78VEFXF6M) runs on
   Saturn and waits for the next rollout dump before linking a completed save.
   Core500 already retains every100-update save. Original100 runs did not save
   model checkpoints; no20-update route-churn history can be reconstructed from
   those runs. Initial/100/200/etc measurements on500 must retain their distinct
   recipe/source provenance.
4. Measure both absolute and relative parameter drift by layer/group, distinguishing
   stored BF16 parameters from FP32 optimizer masters. Read router drift symmetrically;
   direction consistency, expert assignment churn, and policy/auxiliary gradient
   alignment may be more informative than a single norm.
5. Use fixed-token gradient decomposition and then a controlled auxiliary-loss-zero
   training comparison if warranted. Specify whether both balance and router z-loss
   are disabled; disabling only one does not remove the other. Replicated seeds and
   controlled serving numerical settings are needed for causal attribution.
