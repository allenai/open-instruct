# Successive-batch Core scorer qualification

This experiment compares Core290d2ca with isolated candidate5bfa0618f, using immutable
image01M279KKMX3RGXB6AYB273DE3C. The candidate changes only forward SwiGLU row capacity
specialization and dynamic launch-grid stride; it retains the existing arithmetic modes.
No active training runtime pin is changed.

Both arms load the original initial SFT weights and retained Core100 rollout batches5
through9. Each batch is scored twice, retaining caches across successive batches within
an arm. Parent and candidate run in separate two-rank EP2 processes, with separate full
Core source copies and rank-private Triton/Inductor caches. Both use the same installed
OI actor and recipe. No SGLang server, reward generation or optimizer update runs.

The committed patch is applied with zero fuzz to the candidate source copy. Reports
verify kernel hashes, all other Core Python source hashes, OI module hashes, recipe
arguments, input-file hashes and rank partitions. Cross-arm response logprobs must match
bit for bit. Timings/JIT misses/artifact writes and numerical residuals are retained when
numerical checks fail. Raw score tensors remain in the run output and are included in
Beaker results; they contain logprobs only, not response text.

Launch through the standard clean-commit wrapper:

```bash
MILES_EXISTING_IMAGE=01M279KKMX3RGXB6AYB273DE3C \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_core_score_variants.sh
```

The job requests two Holmes GPUs, urgent priority, a30-minute minimum and90-minute
timeout. Default output is the previously unused `score-variants-20260911-v1` directory
under the original100-update campaign. Existing directories are rejected.

This measures repeated compilation under changing retained token batches at initial
weights. It does not establish optimizer/learning parity, reconstruct historical trained
weights or measure Ray ingress. FLA specialization remains separate from the candidate's
SwiGLU change. The earlier identical-batch profile and isolated4090 kernel probe motivated
this experiment; a full-model result is required before promoting the optimization.
