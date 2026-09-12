# Sequence packing in the Core trainer

Experimental feature branch `robertb/miles-sequence-packing`, based on project
primary `016318180`. GPU qualification is pending. The primary branch's defaults
have not been changed by this branch.

```toml
[trainer]
micro_batch_size = 1
sequence_packing = true
packing_max_tokens = 6144
```

`micro_batch_size=1` is one concatenated row containing several complete samples.
The pack budget defaults to Core's maximum individual sequence length and must
cover it. A larger budget increases the trainer forward capacity, not the rollout
context limit. Disabling packing restores the previous unpadded one-sample path;
omit `packing_max_tokens` when disabling. The structured CLI maps both trainer
fields to Core options. MILES receives `qkv_format=thd` for response-logit slicing.

The packer greedily combines consecutive samples within each optimizer batch. It
does not reorder, split, truncate, add padding, change GRPO groups, or cross policy
updates. Original sample lists retain masks, advantages, reference/behavior log
probabilities, rewards and versions. Each response uses only its own preceding
logits. Core receives document lengths so attention/RoPE, KDA recurrence, and short
convolutions respect boundaries. Every sample retains its own final unscored
router-replay token, including interior samples in a pack.

Each rank first plans locally, then all ranks use the largest pack count and
split packs to that count. Equal sample counts guarantee this needs no empty
forwards. This preserves EP forward/backward collectives and the final gradient
reduction schedule. The same construction runs for standalone/reference scoring
and training, including multiple optimizer steps per collection. The scoring
check still guards the first training update and resume; packing does not make
one optimizer update look like several merely because it has multiple packs.

Router auxiliary loss retains Core pretraining's `local_batch` semantics: it is
computed over the tokens in each packed forward. It is not equivalent to summing
per-response balancing losses. Policy-only gradient comparisons therefore disable
auxiliary coefficients; combined-objective checks require finite, nonzero updates
and correct replay/normalization, not equality to the old auxiliary gradients.
No Core/MILES source change or new auxiliary-loss implementation is required.

Packing events in `training_contract_rank*.jsonl` record samples, packs, real
tokens, maximum pack size and fill fraction. W&B step metrics include pack count,
samples/tokens per pack and rank-zero peak allocated memory. Compare warmed
trainer time and memory on identical samples; raw two-step wall time includes
cold compilation and startup. EP8 throughput and larger pack budgets require
separate measurement after the small gate.

## Validation

Host tests cover pack schedules, identity, replay tails, overflow and CLI mapping.
Pinned-runtime fixed-logit tests compare losses and gradients for token/response
reduction, TIS, KL, scoring skip, interior masks and completely masked responses.

Launch the tiny KDA/full-attention/latent-MoE model gate with:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_sequence_packing.sh
```

It uses two Holmes GPUs, urgent priority, a positive minimum runtime, random
local weights and no external datasets. It exercises EP1/EP2 with recomputation
on/off, fixed replay, document-isolation perturbations, two updates (checked then
skipped scoring), policy-only gradient/Adam comparisons and the combined
objective. Per-rank reports and contracts are retained even on failure. This is
followed by a small real SGLang/Core async exercise once the numerical gate passes.
