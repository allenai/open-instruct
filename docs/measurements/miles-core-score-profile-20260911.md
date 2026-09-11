# Frozen Core100 scorer profile

This diagnostic uses the original immutable Core100 image
`01M26N80T0V9PREQTS87J849P8`, original initial HF weights, EP2 and FA4. It loads
retained rollout5 tokens, reconstructs the actual MILES DP partition, and calls the
unmodified Core `_score`. It performs one first-batch pass, three identical repeats,
and a separate warm CPU/CUDA profiler pass. No optimizer update or SGLang server runs.

Run from a clean committed checkout:

```bash
MILES_EXISTING_IMAGE=01M26N80T0V9PREQTS87J849P8 \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_frozen_core_score_profile.sh
```

The launcher embeds only the diagnostic worker and source-hash manifest; it does not
replace runtime libraries. Startup rejects mismatching actor, model factory, data,
original configuration, or SwiGLU sources. The image override is explicit, requires
an immutable Beaker image ID, and retains the wrapper's clean-checkout requirement.
The allocation requests two Holmes GPUs,30-minute minimum,60-minute timeout.

Reports and traces remain under the original campaign's
`score-profile-20260911-v1/`; the path must not already exist. Compact rank reports
are copied to Beaker results even when the numerical gate rejects changed/nonfinite
repeated scores. Raw Chrome traces stay on WEKA. Reports contain token counts and
score residuals, not prompt/response text.

`wall_seconds` includes the complete scorer call and final synchronization.
`cuda_stream_elapsed_seconds` includes stream idle time; it is not GPU busy time.
JIT in-memory misses may load existing disk artifacts. Separate cache-write counts
and extensions distinguish artifact creation; neither counter alone is an exact
count of unique kernel compilations. Rank-private caches prevent one rank borrowing
the other's compilation. Model initialization and filesystem/DP input conversion
are outside the score window. This does not measure Ray object-store ingress or
reconstruct trained step5 weights. Identical token shapes and initial weights suffice
for a bounded compilation hypothesis test; they do not reconstruct changing inputs and
trained weights from the historical run. The profiler pass is excluded from ordinary warm means.

Local validation: nine CPU tests cover frozen-image/topology restrictions, embedded
source payloads, wrapper dirty/alias rejection, residual failures and independent JIT
miss/cache-write accounting. The exact old image passed its original argument parser
and retained-sample conversion/splitting API with sixteen synthetic samples. An actual
4090 smoke verified all frozen module hashes and the observer on the old SwiGLU kernel:
rows37 first call0.292s with one JIT miss and cubin write; identical repeat0.000163s
with no miss/write; changed rows41 call0.0615s with a new miss and cubin write.
[Raw kernel smoke evidence](miles-core-score-kernel-20260911.json) confirms specialization
and observer behavior, not the cause or magnitude of the full-model scoring interval.

## Full-model EP2 result

[Beaker experiment](https://beaker.org/ex/01M279F38FPRHMTYQQYWQ17GSN) completed with
exit0 on September11 at04:03:30UTC. [Compact measurements](miles-core-score-profile-20260911/summary.json)
and complete [rank0](miles-core-score-profile-20260911/rank0.json) /
[rank1](miles-core-score-profile-20260911/rank1.json) reports retain all JIT keys,
constexprs, input lengths and frozen source hashes. There were **zero optimizer updates**.
All repeated response logprobs were finite and bitwise identical to the cold pass
(8,882 response tokens on rank0;9,635 on rank1).

| Pass | Maximum rank wall time (s) | Rank0 / rank1 JIT misses | Rank0 / rank1 cubin writes |
|---|---:|---:|---:|
| Cold |230.4578 |362 /327 |362 /327 |
| Warm1 |1.2491 |0 /0 |0 /0 |
| Warm2 |1.2349 |0 /0 |0 /0 |
| Warm3 |1.2456 |0 /0 |0 /0 |
| Separate instrumented diagnostic |1.7519 |0 /0 |0 /0 |

The ordinary warm mean is **1.2432s**. Model construction/loading took108.22/109.12s
per rank and is outside these intervals. Cold JIT calls consumed134.59/127.25s of
rank-local host wall time. SwiGLU accounted for151/150 specializations and41.38/41.40s;
FLA kernels accounted for the remaining211/177 specializations and93.21/85.85s.
Generated `.ptx` and `.cubin` writes accompany every observed cold miss. JIT durations
must not be added across ranks or subtracted from distributed elapsed time: ranks can
compile concurrently or wait for each other at collectives.

This establishes expensive cold shape specialization and a cheap identical-batch warm
scorer in the frozen runtime. It does **not** prove that all44.33s of historical
collection-to-score-contract time was compilation. That boundary also includes ingress
and checks, and training sees new response lengths/routing on successive batches.
The next attribution measurement should score successive retained batches in one process
while retaining caches and counting new specializations. Optimization should preserve
score equivalence and separate the Core SwiGLU shape key from FLA kernel specialization.
