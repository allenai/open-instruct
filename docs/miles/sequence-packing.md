# Sequence packing in the Core trainer

Packed full-attention models require an attention backend that supports document
boundaries, such as `core.attention_backend="flash_4"`. The `dev` and `small`
examples use Torch attention without packing; enabling packing in a copy also
requires changing that backend. Torch attention rejects intra-document masking.

Basic sequence packing is incorporated into project primary branches
`robertb/miles-olmo-core` (Open Instruct) and `robertb/miles-rl-adapter` (Core).
The implementation was qualified on `robertb/miles-sequence-packing`, originally
based on `016318180`. The small GPU numerical gate and live async retained-data
audit passed. The recommended async example enables packing; the Core option
remains opt-in for other configurations.

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
No new auxiliary-loss implementation is required. The companion Core branch fixes
the FlashAttention 4 variable-length call to bind sequence metadata by keyword;
the pinned API inserts an optional `qv` argument ahead of that metadata.

Packing events in `training_contract_rank*.jsonl` record samples, packs, real
tokens, maximum pack size and fill fraction. W&B step metrics include pack count,
samples/tokens per pack and rank-zero peak allocated memory. Compare warmed
trainer time and memory on identical samples; raw two-step wall time includes
cold compilation and startup. EP8 throughput and larger pack budgets require
separate measurement after the small gate. Packing reduces the number of forwards
and can improve kernel utilization; the existing unpadded path already processes
only real tokens. It does not eliminate variable expert row counts or replace
the dynamic-row SwiGLU specialization fix.

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
objective. Per-rank reports and contracts are retained even on failure. This was followed by a passing small real SGLang/Core async exercise and
independent retained-data audit.


Numerical results and run identities are recorded in
[the measurement notes](measurements/sequence-packing-20260912/README.md).
The small real-model follow-up is configured in
`configs/miles/qualification/sequence-packing.toml`: EP2 plus one TP1 SGLang engine,
three async updates, 8 prompts × 2 responses, replay/recomputation, and a 4096-token
pack budget. This tests plumbing, not GSM8K learning with its short generation cap.

## Replay-informed expert-aware packing (experimental)

Set `trainer.expert_balanced_packing=true` to reorder complete optimizer batches
before MILES partitions samples by rank. The first implementation targets
**multiple complete expert sets**: trainer world size must exceed the expert
parallel degree, and that degree must exceed one. For example, four trainer GPUs
with EP2 provide two complete expert sets. Set `router_aux_loss_weight=0`, enable
sequence packing and rollout routing replay, and use the normal Olmo3MoE HF
configuration. The z-loss coefficient may remain unchanged.

The producer uses recorded expert IDs to place samples with complementary loads
in the same EP group's dispatches. It does not change any tokens, expert IDs,
prompt identities, rewards, policy versions, or optimizer-step membership.
Candidates use arrival and length-based rows, with greedy expert-group placement.
Each candidate is scored through the trainer's exact stride partition, consecutive
packer, and **world-wide** pack-count equalization. Similar lengths do not imply
matching boundaries. The original order wins ties and is retained unless a
candidate improves the measured schedule without increasing pack count, mean or
maximum dispatch skew, maximum destination load, or the absolute-work proxy.
These guarantees concern the counted layers and are not throughput guarantees.

`trainer.expert_balance_layer_stride=1` counts every routed layer (dense layers
are excluded). Larger values sample routed layers and reduce histogram work;
all prediction and trainer metrics then describe only that subset. Histograms
include the trainer's synthetic final replay row separately for every document.
The default-off native arguments are unchanged.

The managed producer callback runs before reward normalization. It requires
original `group_index` and unique sample identities; pinned MILES uses those IDs,
not post-permutation adjacency, for GRPO normalization. Missing/malformed routes,
compact/multi-turn rollouts, dynamic global batch sizes, alternative partitioning,
custom reward/conversion callbacks and conflicting sample filters are rejected.
Any incomplete trailing optimizer block remains untouched for normal MILES trimming.
No MILES or OLMo-core source patch is needed.

Producer `expert_schedule` JSON events record before/after predictions and total
planning time. Trainer `expert_balance` contract events count the actual packs;
W&B exposes `packing/expert_dispatch_skew_mean` and
`packing/expert_dispatch_skew_max`. The work proxy sums the busiest destination
across groups at each pack/layer; it is a count proxy, not predicted wall time.

The dedicated qualification allocates four GPUs and exercises EP2, replay,
per-sample scores, two policy-only updates, full gradients/Adam state, and
activation recomputation on/off:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_expert_schedule.sh
```

A passing fixed-input numerical gate does not establish throughput improvement
or learning quality on a heterogeneous production workload.
