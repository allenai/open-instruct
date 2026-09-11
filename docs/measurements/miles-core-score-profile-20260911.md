# Frozen Core100 scorer profiling protocol

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
for a bounded compilation hypothesis test; final attribution needs the full-model
measurements. The profiler pass is excluded from ordinary warm means.

Local validation: nine CPU tests cover frozen-image/topology restrictions, embedded
source payloads, wrapper dirty/alias rejection, residual failures and independent JIT
miss/cache-write accounting. The exact old image passed its original argument parser
and retained-sample conversion/splitting API with sixteen synthetic samples. An actual
4090 smoke verified all frozen module hashes and the observer on the old SwiGLU kernel:
rows37 first call0.292s with one JIT miss and cubin write; identical repeat0.000163s
with no miss/write; changed rows41 call0.0615s with a new miss and cubin write.
[Raw kernel smoke evidence](miles-core-score-kernel-20260911.json) confirms specialization
and observer behavior, not the cause or magnitude of the full-model scoring interval.
