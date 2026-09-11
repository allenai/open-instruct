# Hero support in the MILES / OLMo-core integration

The hero work is isolated on `robertb/miles-hero-support`. The older SFT GSM8K
trial continues with its original image and checkpoint.

## Architecture and source lineage

The non-EMO small hero has 16 blocks (14 KDA and two full-attention blocks),
hidden width 1024, latent width 512, 512 experts and top-16 routing. It has
12,496,339,072 parameters, approximately 794 million active per token. These are
checkpoint properties, not constants in the adapter.

The audited training revision is `89bf37a87d955b8ff8a76ac11df6dd3bec976d30`
on `allenai/OLMo-core:codex/small-hero-20260907`. The support port uses
`codex/small-hero-hf-20260909` at
`b1fd2c9746e88baeb20e372bdca340d788d0f7e5`, which includes the later HF
implementation of per-head Q/K gains and scalable softmax. The original
gdn2 adapter patch is not directly applicable to this branch; conversion and
router changes must be reconciled and retested.

## What changes

| Piece | Hero support |
| --- | --- |
| MILES rollout/training loop | Reuse the existing Core actor hookup, rewards, objectives, rollout management and policy versions. |
| Core model construction | Read geometry, KDA/full-attention layout, dense blocks, latent MoE, per-head Q/K gains and scalable softmax from config. Preserve router settings. |
| Weight conversion | Preserve the two-dimensional Q/K gain tensors and learned `self_attn.ssmax_scale`. Detect native dense-MLP packing and keep strict tensor coverage. |
| SGLang | Add per-head normalization and scalable softmax using absolute request positions, including chunked prefill and decode. |
| Live weight publication | Keep the existing named-tensor/bucket transport. Include the new parameters and copy into existing storage so captured graphs retain valid pointers. |

Full-attention scalable softmax multiplies Q by
`log(position + 1) * ssmax_scale`, preserving the HF implementation's dtype and
rounding order. Positions must be local to each request, not indices in a
flattened batch. The usual attention `1/sqrt(head_dim)` factor still applies.

## Conversion qualification

The real checkpoint inputs are read without modifying either source:

- Native: `/weka/olmo-3p5-checkpoints/production-hero-small/olmo35-small-hero-20260907/olmo35-small-hero-20260907-non-emo/step38000`
- HF reference: `/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/non-emo/step38000/hf`

`scripts/miles/validate_hero_conversion.py` performs two exhaustive checks:

1. Build the saved native architecture and load its model weights, including
   flattened optimizer-backed master weights when that is the checkpoint format.
   Stream the HF mapping and compare every tensor to the reference export.
2. Build the Core adapter from the HF config, import the HF tensors, then stream
   them back and compare every tensor again.

Both passes reject missing, duplicate, extra, wrong-shaped and nonfinite weights.
Values must match exactly after the explicitly recorded export dtype conversion.
The first pass may trim only embedding/LM-head vocabulary padding, using the saved
tokenizer and model vocabulary sizes. Reports record tensor counts, bytes, content
hashes, config hashes, attention parameter shapes, dtype casts and packing metadata.
The gate does not allocate optimizer moments or claim generation parity.

Tiny tests use multiple model widths and expert counts, distinct non-unit gains
and scales, and the real distributed-checkpoint reader. The real checkpoint audit
uses a CPU worker on Saturn with WEKA mounted, urgent priority and a minimum runtime:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_hero_conversion.sh
```

Source changes must be committed before launching. Native and HF paths can be
overridden with `--native` and `--hf`; the geometry remains config-driven.

## Subsequent gates

Conversion alone does not qualify RL. Before a hero RL run, check tiny cached
generation against the HF reference, mixed request lengths, chunked prefill,
CUDA graph replay with changing positions, and publication of changed gains and
scales. Then compare real-checkpoint Core and rollout log probabilities and run a
short training trial. Start with TP1 serving; larger TP and distributed training
need their own qualification. EMO is outside this first non-EMO target.

The existing SFT run is a separate experiment:
[100-update GSM8K trial](https://beaker.org/ex/01M26P6XX6SN886DCVZ68WMQK2),
[W&B](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/un6so0cr).
Its initial held-out score was 97/128, and the 20-update score was 101/128.
Those measurements concern the older SFT architecture and do not qualify hero.
