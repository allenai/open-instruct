# Hero support in the MILES / OLMo-core integration

Both corrected-tokenizer hero SFT checkpoints passed a bounded
[four-update MILES mechanics check](measurements/hero-rl-smoke-20260924.md) with
H100 EP4 training, TP1 automatic fused-rounding serving, nonzero gradients and
exact live weight publication. This establishes the recorded short barrier path;
learning quality, mixed-policy refresh, long context and save/resume remain
unqualified. The earlier HF-reference probability failures below are retained as
historical evidence and are not being relabeled as passes. The new runs kept the
Core/behavior mean-gap guard at 0.05. See the [support matrix](feature-parity.md).

The [September 23 corrected-tokenizer SFT audit](measurements/hero-sft-20260923.md)
locates paired 4T EMO/non-EMO pretraining lineages and their full SFT configs.
Both SFT checkpoints disable EMO and pass exact architecture/weight conversion
checks. H100 and B300 inference match the sampled greedy tokens but fail the
unchanged probability gate in that earlier study, which deferred the training
trial until the numerical investigation. These are separate checkpoints
from the historical base-model qualification below.

The [SFT numerical isolation](measurements/hero-numerics-20260923.md) finds exact
HF/SGLang routing on identical inputs across all tested layers and prefixes.
Routed-expert, full-attention and smaller RMSNorm arithmetic differences explain
the observed forward mismatch; diagnostic HF substitutions remove it. That
diagnostic predates the fused-rounding implementation and the bounded training
exercise above; its historical failed gate result is unchanged.

The [direct Core/HF fidelity investigation](measurements/hero-core-fidelity-20260923.md)
finds exact agreement between the HF Core-layout reference, current Core scoring,
and Core gradient-enabled forward for both SFT checkpoints on the tested inputs.
Ordinary HF execution differs, despite faithful exported tensors. Partial serving
rounding/norm changes improve some cases but do not qualify the checkpoints;
Core's actual training policy is the recommended serving comparison target.

The [serving-mode guide](core-compatible-serving.md) documents automatic fused
rounding for the qualified hero profile and its opt-out. The separate full
reference (`OLMO_SGLANG_CORE_COMPAT=1`) remains opt-in and requires TP1 BF16
serving with CUDA graphs disabled. This option does not by itself qualify
full hero RL or change the historical gate results below.

## Architecture and source lineage

The non-EMO small hero has 16 blocks (14 KDA and two full-attention blocks),
hidden width 1024, latent width 512, 512 experts and top-16 routing. It has
12,496,339,072 parameters, approximately 794 million active per token. These are
checkpoint properties, not constants in the adapter.

The audited training revision is `89bf37a87d955b8ff8a76ac11df6dd3bec976d30`
on `allenai/OLMo-core:codex/small-hero-20260907`. The support port uses
`codex/small-hero-hf-20260909` at
`b1fd2c9746e88baeb20e372bdca340d788d0f7e5`, which includes the later HF
implementation of per-head Q/K gains and scalable softmax. The adapter port reconciles the earlier gdn2 conversion/router changes with this
lineage. The runtime lock pins the base and the reviewed patch.

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

The initial step38000 native paths were absent: production had retained newer
steps, and the archived native copy had been cleaned up. The read-only
[inventory job](https://beaker.org/ex/01M26Y0F5XD5RPVFP28V52YX3K) found native
steps 72000–80500 and an HF export directory for step75500. The conversion trial
therefore uses the matched step75500 pair, without modifying either source:

- Native: `/weka/olmo-3p5-checkpoints/production-hero-small/olmo35-small-hero-20260907/olmo35-small-hero-20260907-non-emo/step75500`
- HF reference: `/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/non-emo/step75500/hf`

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

## Standard Olmo 3 trainer

`open_instruct/miles/training/models.py` is the shared actor-facing façade. The specialized
OLMoDDP factory, optimizer, HF import/export, native checkpoint methods and router
replay live in `moe_models.py`. Dense Olmo 3 uses `standard_models.py`, Core's
`TransformerTrainModule`, AdamW and the standard FSDP path where applicable.
Disabled replay does not load a backend; dense configurations reject replay and
expert parallelism before allocating model weights.

Actual tiny Olmo 3 full/sliding models pass HF logit checks and exact weight
roundtrips. Both also pass a real MILES loss/update and exact next-update restore
check on a local GPU, alongside Qwen3, KDA and latent KDA. The checks found and
fixed a duplicated sliding-window decrement on this newer Core lineage. Full-size dense lifecycle evidence is described in the [dense guide](olmo3-pre-rl.md);
hero qualification does not transfer to that model, or vice versa.

## Current local evidence

- Core port: 33 focused factory/conversion/replay/objective/attention tests.
- Existing integration: 91 CPU contracts on hero Core before the backend split.
- Backend split: 27 conversion/factory checks and five GPU update/resume cases.
- Replay isolation: 17 focused cases, including distributed malformed-input checks.
- Conversion gate: 24 cases including shard-index mismatch, competing native/master
  copies, actual flattened-master checkpoint reads, and multiple geometries.
- SGLang: 85 tests plus tiny hybrid TP1 generation. All eight greedy tokens matched
  HF; maximum conditional logprob error 0.02998. Chunked prefill and decode graphs
  passed. Separate live gain/scale buckets changed outputs, then matched a fresh
  changed-checkpoint engine with identical tokens and zero logprob difference.

These are scoped results, not full hero RL acceptance. The pinned runtime lock
records the exact Core and SGLang source revisions and patch checksums. SGLang's
runtime patch covers `src`, `tools` and `docs`, because the compiled base image
omits original test files; the full regression suite is retained in its local
source commit.

The optional three-way tiny check also exercises Core's native factory, imported
weights and forced-prefix logits. With aligned tiny expert widths it passed:
Core max/mean logprob error 0.03333/0.00358; SGLang max error 0.04115; all eight
HF greedy tokens agreed in both serving modes. This explicitly uses semantic
reference attention/KDA execution, not the production optimized training kernels.
The full-checkpoint serving launcher enables the same comparison with a preset
0.1 maximum absolute logprob error gate and records actual errors.

## Full checkpoint weight gate

[The step75500 conversion job](https://beaker.org/ex/01M26YP78T0JJ545H914AEYDDZ)
finished with exit zero. All 23,441 HF tensors matched native FP32 masters after
the expected BF16 export cast and embedding/head vocabulary trim. HF import
through the adapter and re-export produced the identical tensor-content digest.
The [machine-readable report](measurements/hero-conversion-20260910.json)
records architecture, source pins, CPU-only execution overrides, 579 seconds
elapsed and 72 GiB peak RSS. No optimizer moments were loaded.

[Full-checkpoint scoring](https://beaker.org/ex/01M26ZBDKPDBRJ03TYT6V2GYRJ)
finished with exit one because both Core and SGLang exceeded the predeclared
0.1 log-probability limit. All eight greedy tokens matched HF in both serving
modes. Core maximum full-vocabulary error was 0.5473 (mean 0.08255); SGLang
maximum top-20 conditional error was 0.5619, with sampled cached-decode error
0.08974. These measure different subsets and are not equivalent statistics.
[The retained report](measurements/hero-serving-20260910.json) records
the failed gate. This failure motivated the subsequent layerwise investigation and Core-based
checks linked above; token agreement did not override the failed probability check.
