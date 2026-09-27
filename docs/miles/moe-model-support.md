# Hybrid MoE checkpoint support

The specialized Core adapter supports config-driven KDA/full-attention layouts,
latent MoE, per-head Q/K gains and scalable softmax. The measured 12.5B profile
has 16 blocks (14 KDA and two full-attention), hidden width 1024, latent width
512, 512 experts and top-16 routing. These are checkpoint properties, not
constants in the adapter.

Use the sources in the [runtime lock](architecture.md#runtime-sources-and-images).
The adapter lives in MILES; OLMo-core supplies model construction and conversion,
and olmo-sglang supplies the serving implementation. See
[serving modes](core-compatible-serving.md) for compatible arithmetic and limits.

## Conversion checks

`scripts/miles/validate_hero_conversion.py` accepts explicit `--native` and `--hf`
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

Tiny runtime tests exercise multiple widths and expert counts, distinct non-unit
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

## Qualification limits

Two corrected-tokenizer SFT checkpoints passed a four-update synchronous barrier
mechanics check with H100 EP4 training, TP1 fused-rounding serving, nonzero
gradients and exact live publication. That result does not qualify learning
quality, mixed-policy refresh, long context or save/resume for this profile.

Earlier full-checkpoint HF-reference probability gates failed despite matching
greedy tokens. Later Core-reference investigation and bounded training checks do
not retroactively turn those failures into passes. Use Core's actual scoring
policy for rollout comparisons and retain both token and probability checks.

Before extending the profile, exercise cached generation, mixed request lengths,
chunked prefill, graph replay with changing positions and publication of changed
gains/scales. Larger tensor parallelism and different model geometries need their
own qualification. Dense Olmo 3 uses a separate FSDP adapter; results do not
transfer between the two paths.

The [archived support record](https://github.com/allenai/open-instruct/blob/7a477910405a65d914b48096f83c13a6c61a60ad/docs/miles/hero-support.md)
retains checkpoint lineages, source revisions, measurements and experiment IDs.
See the [support matrix](feature-parity.md) for the other supported paths.
