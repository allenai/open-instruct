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

**Experimental, opt-in, and disabled by default.** The planner has passed
correctness qualification on the tested configurations. Net throughput benefit
and effects on learning across workloads and topologies remain unestablished;
planning overhead can offset the trainer-time savings. Measure total planning and
training time on your workload before enabling it for a production run.

The implementation is already incorporated into the project working branch
`robertb/miles-olmo-core`, including bounded swap search. All maintained examples
leave it disabled. This option is separate from ordinary sequence packing, which
can remain enabled while expert-aware scheduling is off.

Set `trainer.expert_balanced_packing=true` to reorder complete optimizer batches
before MILES partitions samples by rank. The first implementation targets
**multiple complete expert sets**: trainer world size must exceed the expert
parallel degree, and that degree must exceed one. For example, four trainer GPUs
with EP2 provide two complete expert sets. Set `router_aux_loss_weight=0`, enable
sequence packing and rollout routing replay, and use the normal Olmo3MoE HF
configuration. The z-loss coefficient may remain unchanged.
`miles.balance_data=true` is rejected with expert-aware packing: MILES length
balancing changes the stride partition that the expert planner assumes. Enable
only one of these two planners.

The producer uses recorded expert IDs to place samples with complementary loads
in the same EP group's dispatches. It does not change any tokens, expert IDs,
prompt identities, rewards, policy versions, or optimizer-step membership.
Candidates start with arrival and length-based greedy placement, then a bounded
swap search starts from arrival order. Three quarters of proposals target heavy
contributors to currently overloaded destinations and complementary samples;
the remainder explore random swaps. Half of the targeted proposals favor nearby
lengths, but similar lengths are **not** assumed to preserve pack membership.
Each proposed swap repacks its affected columns. If the world-wide pack count
changes, all columns are re-equalized. When final position membership is exactly
unchanged, only the affected dispatch counts need updating.

Search minimizes the sum of three arrival-normalized stage-work proxies:

- Expert work: sum over packs/layers of the largest destination load across all
  expert replicas.
- Token-linear attention work: sum over packs of the largest rank's total tokens.
- Quadratic attention work: sum over packs of the largest rank's sum of squared
  **document** lengths, not the square of concatenated pack length.

The equal search weights are a heuristic, not calibrated kernel times. On an
exact-objective plateau, a strictly better log-sum-exp surrogate can advance the
search. The returned candidate is tracked separately and cannot worsen either
attention proxy, expert work, pack count, mean/maximum within-replica dispatch
skew, or maximum destination load relative to the retained greedy/arrival
baseline. Search states may temporarily violate that final guard. Rejection
counts by metric expose this distinction. These are aggregate count guarantees
for the selected layers, not per-pack guarantees or throughput predictions.

`trainer.expert_balance_search_proposals=1024` bounds proposal attempts per
optimizer block; zero retains only greedy placement. The default
`trainer.expert_balance_search_seconds=0.25` allowance is divided across complete
blocks in a collection. Histogram construction, greedy scoring and logging are
outside that allowance; an in-progress proposal may finish after the deadline.
Search also stops after 256 attempts without an accepted move or when all work
lower bounds are reached. The bounds relax packing/indivisible-sample constraints
and do not certify an optimal partition except when attained.

A stable seed is derived from the block's original sample IDs. A fixed attempt
budget is reproducible; a deadline can truncate at a machine-dependent point.
Logs retain the seed, completed attempts, selected local-index permutation,
acceptance counts, scoring paths, bounds and stop reason. The current managed
hook still runs synchronously at collection drain; this pass does not introduce
an asynchronous planning actor or pre-arrival histogram transport.

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

### Observed scope: September 2026

The [four-GPU EP2 numerical qualification](https://beaker.org/ex/01M2YQ3QAHHT34W6C0XK5XNFWX)
passed score, replay, gradient and Adam-state comparisons on fixed inputs, within
the existing numerical tolerances. These are correctness checks, not performance
or learning guarantees.

The completed 100-update mixed-workload comparison used eight trainer GPUs
(DP4/EP2), seven policy engines, one judge, and both router auxiliaries disabled:
[packing off](https://beaker.org/ex/01M2ZVH6TQVS037NPD2CWKWXBH) versus
[packing on](https://beaker.org/ex/01M308PE9GNF05CHHETTCBZMRP). All 100 treatment
updates matched predicted dispatch counts on all eight ranks, and periodic
standalone/training scoring checks passed. Held-out results showed no consistent
learning advantage or regression; one trajectory per arm and small evaluation
panels do not establish equivalence.

During updates 61–100, timed trainer work was 18.50 minutes off versus 15.10
minutes on, while treatment planning took another 3.08 minutes. Generation
dominated elapsed time. The control resumed at update 40 and treatment ran fresh,
with independent asynchronous samples, so these timings are descriptive rather
than a matched-input causal estimate. Keep this feature off by default until its
net benefit is measured for the intended topology, batch and workload.


The CPU benchmark accepts a routing-panel JSON file with per-document expert
histograms (no GPU or new generation required):

```bash
python -m scripts.miles.benchmark_expert_search PANEL.json OUTPUT.json
```

It compares greedy-only and bounded search on task/general panels, verifies each
returned result with a full rescore, and records the input checksum. Panel
histograms need not reproduce a live rollout's final synthetic replay rows;
these are offline scheduling measurements on the supplied counts.

Related work: [ReLibra](https://arxiv.org/html/2605.08639v1) uses incremental
swap search and an LSE surrogate for expert placement, followed by sample-locality
optimization. [ForeMoE](https://arxiv.org/html/2606.11867v1) schedules expert
placement/replication using foreseen routing. [RoutePack](https://arxiv.org/html/2608.12146v1)
explicitly couples attention work with expert-aware packing. This implementation
keeps expert placement and routing fixed; it does not claim novelty for replay-aware
packing, calibrated communication costs, or their reported speedups.
