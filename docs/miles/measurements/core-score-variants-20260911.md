# Successive-batch Core scorer qualification

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

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

Submitted September11 at04:29:08UTC from clean committed OI source2d945483a:
[Beaker experiment](https://beaker.org/ex/01M27BKYTF9N7JMYBTKAV0HS9A). The immutable
image was reused explicitly through the standard wrapper. Eighteen focused tests passed
in that pinned image; the final nine comparator/launcher cases passed again after recipe
argument validation. A local source-copy preflight applied the exact patch with zero fuzz
and verified both expected kernel hashes. Results below passed.

## Result

The job exited0 at04:54:34UTC. [Complete compact evidence](core-score-variants-20260911/summary.json)
retains all artifact hashes, per-rank source fingerprints, JIT counts and timings.
[Exact comparison](core-score-variants-20260911/comparison.json) passed on all ten
rank/batch combinations and154,531 response logprobs: maximum and mean absolute differences
were zero. Repeated scores within each arm also matched exactly. All668 other Core
Python files had the same aggregate hash; OI modules, recipe arguments, input files and
rank partitions matched. Neither arm performed an optimizer update.

| Retained batch | Parent first pass (s) | Candidate first pass (s) | Parent repeat (s) | Candidate repeat (s) |
|---|---:|---:|---:|---:|
|5, initially cold |199.104 |125.199 |1.193 |1.240 |
|6 |85.051 |35.869 |1.243 |1.239 |
|7 |58.879 |10.384 |1.248 |1.232 |
|8 |47.153 |1.862 |1.253 |1.197 |
|9 |62.048 |12.746 |1.268 |1.249 |

Each entry is the maximum rank wall time of `_score`, including final synchronization.
The four subsequent first-pass batches averaged63.283s on the parent and15.215s on the
candidate:75.96% less scorer time, or4.16× for this specific retained-batch diagnostic.
This is not an end-to-end training speedup or a throughput prediction for the active runs.

Parent SwiGLU generated150/151 variants on the initial batch and142–152 additional
variants per rank on every subsequent batch. The candidate generated exactly one SwiGLU
variant per rank initially and **zero** thereafter. Parent per-rank SwiGLU JIT calls
consumed about41–43s per subsequent batch. Candidate remaining misses were in FLA kernels;
for example batch7 rank0 had zero misses but waited for rank1, which compiled32 kernels.
Therefore per-rank compile durations must not be summed or mechanically subtracted from
distributed elapsed time. All observed misses had corresponding generated cubin writes.

This establishes recurring capacity specialization as a substantial, removable scorer
cost. Both arms used initial weights and fixed retained samples, not successive learned
policies. Their order was parent then candidate on one allocation, without randomized
repetition, so broader performance claims remain unqualified. The active500 learning
runs keep their existing pinned kernels. The isolated candidate is available for a
separately reviewed integration; the unrelated pairwise exact-gradient test failure
reproduces on its unchanged parent and remains documented with local kernel evidence.
