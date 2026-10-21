# Hybrid MoE checkpoint support

The specialized Core adapter supports config-driven KDA/full-attention layouts,
latent MoE, per-head Q/K gains and scalable softmax. The 12.5B profile
has 16 blocks (14 KDA and two full-attention), hidden width 1024, latent width
512, 512 experts and top-16 routing. These are checkpoint properties, not
constants in the adapter.

Use the sources in the [runtime lock](architecture.md#runtime-sources-and-images).
The adapter lives in MILES; OLMo-core supplies model construction and conversion,
and olmo-sglang supplies the serving implementation. See
[serving modes](core-compatible-serving.md) for compatible arithmetic and limits.

## Conversion checks

`scripts/miles/validate_moe_conversion.py` accepts explicit `--native` and `--hf`
paths and performs two exhaustive checks:

1. Load the saved native architecture and model weights, including flattened
   optimizer-backed master weights where applicable. Stream the HF mapping and
   compare every tensor to the reference export.
2. Build the Core adapter from the HF configuration, import its tensors, then
   stream them back and compare every tensor again.

Both passes reject missing, duplicate, extra, wrong-shaped and nonfinite weights.
Values must match exactly after the recorded export dtype conversion. The native
pass may trim only embedding/LM-head vocabulary padding, using the saved sizes.
Reports retain tensor/config hashes, dtype casts and packing metadata. This check
loads no optimizer moments and does not establish generation parity.

Tiny runtime tests cover multiple widths and expert counts, distinct non-unit
attention gains/scales and the real distributed-checkpoint reader. For a full
checkpoint, use a CPU-only unallocated job with the required WEKA mounts: omit
`context.minRuntime`, try Saturn first, and use Jupiter after checking scheduler
events and canceling an unschedulable attempt. Follow the
[launch procedure](launching.md) and commit source changes before submission.

## Attention and publication contracts

Conversion preserves the two-dimensional Q/K gain tensors and learned
`self_attn.ssmax_scale`, with strict tensor coverage and native dense-MLP packing
detection. Scalable softmax multiplies Q by `log(position + 1) * ssmax_scale` in
the reference dtype and rounding order; the usual `1/sqrt(head_dim)` factor still
applies. Positions must be local to each request, including chunked prefill and
decode. Publication includes these parameters and copies into existing storage
so captured graphs retain valid pointers.

## Current limits

Synchronous barrier publication with EP4 training and TP1 fused-rounding serving
is supported for this profile. Mixed-policy refresh, long context and save/resume
are untested for it.

Matching greedy tokens does not imply matching probabilities against an HF
reference. Use Core's actual scoring policy for rollout comparisons and retain
both token and probability checks.

Before extending the profile, test cached generation, mixed request lengths,
chunked prefill, graph replay with changing positions and publication of changed
gains/scales. Larger tensor parallelism and different model geometries need their
own checks. Dense Olmo 3 uses a separate FSDP adapter; results do not transfer
between the two paths.

See the [support matrix](feature-parity.md) for the other supported paths.
