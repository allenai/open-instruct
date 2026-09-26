# MILES with an OLMo-core trainer

Use the [operating guide](index.md) for new GRPO runs and the
[support matrix](feature-parity.md) for qualified boundaries. The adapter replaces
the MILES Megatron trainer actor, while retaining MILES rollout actors, samples,
advantages, PPO loss helpers and transport primitives. Open Instruct supplies
configuration, data/verifiers, lifecycle coordination and retained evidence.

## Trainer hookup

`open_instruct/miles/training/actor.py` implements the trainer actor contract. The
`models.py` facade selects `moe_models.py` for the specialized OLMoDDP MoE path
or `standard_models.py` for dense Core `TransformerTrainModule`/FSDP. MILES calls
the actor for initialization, scoring, training, save/restore and weight export.
The Core objective hook accepts MILES' token log-probability loss and performs
backward, distributed gradient synchronization and optimizer stepping through
Core. It does not select the MILES FSDP trainer or construct a Megatron model.

`driver.py` coordinates collection, optimizer completion, publication, evaluation
and checkpoint boundaries. A policy version advances only after a successful
optimizer step. The async producer attaches behavior versions to samples; the
trainer checks their age again when consuming the batch.

The Core adapter lives on OLMo-core's `robertb/miles-rl-main` branch, based on
Jacob's [production MoE PR #872](https://github.com/allenai/OLMo-core/pull/872)
at `ad28862b5`. The runtime pins `e505356353aa7ce1f6ff83e24d6eb945f463714e`
directly; image builds fetch that commit without applying a Core patch.
The port retains custom objectives, routing replay/count controls, bounded
checkpoint planning and streaming HF interchange, including the inherited
per-head attention and hybrid configuration export support.

Local checks cover model/configuration roundtrips, checkpoint planning, adapter
contracts and a single-GPU hybrid-MoE scoring/backward/optimizer step. Multi-GPU EP,
Blackwell-only paths and full-policy runs still need qualification on this new
base. Earlier measurement reports describe their original source and image pins.
Use the [runtime lock and build procedure](architecture.md#runtime-sources-and-images)
to reproduce the current source.

## Samples, scoring and objectives

Core microbatch size is one. Optional [packing](sequence-packing.md) combines
complete samples in a document-isolated forward without crossing optimizer
partitions. Global batch size must divide collections into whole optimizer
steps and be divisible by trainer world size. Every rank executes the same
number of forwards/backwards. Masks, response boundaries and next-token logits
remain attached to their original sample.

A standalone scoring pass supplies an unchanged trainer-policy anchor when
needed. With one optimizer step per collection, no reference KL and no dropout,
the training forward can supply that anchor. The first update of each process
and periodic checks compare the two forwards before permitting the skip.
See [scoring controls](run-controls.md) for forcing the pass and its tolerance.
Async TIS corrects the behavior-versus-trainer ratio; it does not make stale data
on-policy. Allowed lag is in optimizer steps, including multiple steps in a
collection. Async requires positive lag and resident disaggregated engines.

Core's router auxiliary/z losses are independent of PPO clipping, advantage
normalization and TIS. Replay fixes selected expert IDs; router probabilities
and weights still participate in differentiable computation. Packing retains
Core pretraining's local-batch auxiliary loss over each packed forward's tokens.
It is not a sum of independently balanced response losses.

## Router replay and parallelism

Enable both `use_rollout_routing_replay=true` and `use_miles_router=true`. The
captured IDs travel with each sample through DP distribution and packing. The
adapter aligns replay by document/token/layer and retains it across backward
recomputation. Each sample's final unscored token has an explicit synthetic tail;
that convention is not a claim of Megatron auxiliary-loss equivalence.
Expert-parallel ranks receive the routing information needed by their local
forward; trainer PP/TP/CP greater than one are rejected, not implicitly supported.

[Full-model replay](measurements/core-replay-full-sft-20260911.md) and the later
[packed async restart](measurements/colleague-20260913/README.md) passed their
retained-route audits. A replay match verifies supplied assignments; it does not
measure what the trainer would have selected without replay.

## Checkpoints and publication

SGLang starts from HF config/tokenizer/weights. Core imports those weights into
its native training representation. Updated policies are streamed directly as
serving tensors; no HF directory or Megatron conversion is written per update.
MoE export supports fused expert tensors; final HF exports retain standard
per-expert slices. Dense export gathers parameters through its FSDP backend.
The normal disaggregated transport broadcasts flattened NCCL buckets; colocation
uses IPC. Current full-model starters use mixed-policy `refresh`; low-level
`barrier` and independent [engine drain](engine-drain.md) remain distinct modes.
[Publication contracts](grpo.md#publication-modes) explain historical behavior
probabilities, final-forward replay routes and oldest-token lag checks.

Native checkpoints include model/optimizer/scheduler and rank RNG state. Schema-2
manifests record world size, EP degree and the committed rollout cursor. Restore
validates these before native loading collectives. The data cursor includes
pristine pending prompts; partially generated responses are regenerated after
restart. Serving RNG and exact future generations are not restored. Schema-1
multi-rank checkpoints require explicit migration, and topology changes on
restore are not supported.

A deliberate `debug_exit_after_rollout` stop preserves a resumable workflow.
Final HF export is deferred until the configured collection horizon completes;
the first process must not occupy the final export directory prematurely.
See [model/checkpoint operations](models-and-checkpoints.md).

## Training contract checks

Every real training step now records `training_contract_rankN.jsonl` beneath
`miles.save`, when supplied, and emits the same records to logs. Records include:

- Actual global sample, active-token and model-token counts; policy and auxiliary
  denominators; local accumulation count; response-versus-token reduction mode.
- Local normalized policy objective and weighted auxiliary terms separately.
  Averaging these local objectives across ranks gives the corresponding global
  objective; they are not already global metrics.
- Consumed/published policy versions, LR used/next, completed step and elapsed time.
- Globally reduced probability-error histograms, exact maximum, upper-bin estimates
  of p50/p95/p99, response-position thirds and response-length buckets. A null
  quantile upper bound means the overflow bin (>1.0), not missing observations.

`core.diagnostic_interval=N` additionally measures local gradient norms before
optimizer intake and sampled model updates every N optimizer steps. These are
explicitly not global optimizer norms: expert gradients may still need Core's
EP-MP rescaling, and FP8 stores outside `named_parameters` are outside coverage.
Update samples retain at most 256 values per named parameter, rather than cloning
the model. With `check_weight_update_equal=true`, the driver also checks serving
weights at publication boundaries whose rollout count is divisible by N
(and on initial publication). A fresh run compares its initial
publication against SGLang's original HF snapshot. Periodic and resumed checks
snapshot the **current** serving state, reset tensors, republish the same trainer
version, and compare exactly when quantization tolerance is disabled, over the
non-skipped tensors. This catches missing or inconsistent transfers;
it is not an independent proof of the export mapping for changed weights.
Logprob checks and the initial HF comparison supply separate evidence. The
round trip adds a second full transfer, with `repeated_version=true` in the
publication log; include both transfers when measuring diagnostic overhead.
The async producer remains paused until the check completes.
The interval defaults to zero, leaving these periodic probes disabled. Our training
examples explicitly keep `core.diagnostic_interval=0` and
`check_weight_update_equal=true`: the full weight audit runs at startup, including
resume, and does not run after ordinary training updates. Reserve a positive
interval for qualification or investigating publication correctness. This interval
also controls the extra gradient-norm and sampled-parameter-update diagnostics;
setting it to zero does not disable the separate policy-version, logprob, replay,
or scoring-pass checks. Upstream MILES supplies the optional startup check; the
periodic same-version round trip is an addition in our Core driver.

Runtime failures include non-finite active inputs/rewards, empty effective
batches, inconsistent rank schedules, stale policy versions, non-finite losses,
scheduler/clock disagreement and skipped optimizer steps. All ranks agree on
skip status before any policy clock advances. The existing mean score-drift guard
remains configurable; tail distributions are measured without inventing an
unqualified universal cutoff. A skip aborts the run; this is not transactional
rollback of ranks that already performed an optimizer update.


## Numerical evidence and development checks

The independent fixed-batch reference checks next-token slicing, masks, unequal
lengths, PPO clipping, entropy, KL and loss reduction without reusing the loss
helpers it is testing. Tiny native EP1/EP2 comparisons cover policy-only,
auxiliary-only and combined gradients/Adam state, with recomputation on/off.
These are native consistency checks, not full-model Megatron equality.
[Contract report](measurements/core-contract-final-20260910.json) and
[native EP report](measurements/core-native-ep-20260910.json) retain thresholds
and measured errors. Current test counts belong to each image's qualification.

```bash
# Run inside the pinned image; the GPU suite needs the allocated device(s).
python scripts/miles/check_contract.py /output/contract
# Submit the native EP gate through the committed-image wrapper:
MILES_EXISTING_IMAGE=IMMUTABLE_IMAGE_ID \
  ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_core_contract.sh
```

The [historical local MoE procedure](measurements/implementation-history/core-before-sharing-20260913.md#local-moe-task-and-restart-check)
records the original fixture/debug workflow. Use the current structured examples
for new runs; old trial commands and then-pending gates are not current defaults.

## Router auxiliary objectives

These optional controls change the MoE router's auxiliary losses. They are
independent of the RL policy loss's token-versus-response normalization and of
whether routing replay is enabled. Existing defaults remain unchanged.

| Field under `[core]` | Default | Alternative and meaning |
|---|---|---|
| `router_aux_loss_grouping` | `"pack"` | `"sequence"`: compute balancing statistics separately for each original prompt-plus-response document. |
| `router_aux_loss_reduction` | `"token"` | `"response"`: give each document equal weight instead of weighting by model-token count. |
| `router_z_loss_reduction` | `"token"` | `"response"`: independently give each document equal weight for z-loss. |
| `router_aux_count_source` | `"dispatch"` | `"current"`: use the current forward's router top-k choices to count expert selections for balancing. |
| `router_aux_loss_weight` | `0.01` | Balancing coefficient; `0.0` disables its gradient contribution. |
| `router_z_loss_weight` | `1e-5` | Z-loss coefficient; `0.0` disables its gradient contribution. |

### Grouping, then averaging, then the coefficient

Grouping decides which tokens share an expert-usage histogram. `"pack"` shares
one histogram across the physical forward; `"sequence"` makes a separate histogram
for each original document even when several documents occupy one pack. Neither
choice changes attention boundaries or physical packing.

Averaging decides how each document contributes to the update. With sequence
grouping, suppose two documents have 2 and 6 model tokens and their balancing
penalties are 1 and 3. Token averaging gives `(2*1 + 6*3) / 8 = 2.5`; response
averaging gives `(1 + 3) / 2 = 2`. A balancing coefficient of `0.01` makes their
contributions to the training objective `0.025` and `0.02`, respectively. The same
weights and coefficient scale their gradients.

With pack grouping and response averaging, documents share the pack's expert
histogram, but each document's mean router probabilities has equal weight.
Response averaging does not implicitly select sequence grouping. Z-loss has no
expert histogram; its averaging is selected independently.

Here a document is **the prompt plus its response**, including the final forwarded
token and its synthetic replay assignment. These are model tokens, not only
policy-loss tokens. Padding is excluded. The token and document denominators span
the optimizer update and data-parallel ranks, accounting for gradient averaging;
this is not an average of microbatch means.

### Dispatched versus current counts

`"dispatch"` counts the actual selected experts. With replay active, those are the
replayed assignments. `"current"` instead derives a detached top-k histogram from
this forward's router scores. It changes the balancing statistics only: replayed
expert dispatch, selected expert weights, load metrics and z-loss keep their
existing meanings. Gradients still flow through router probabilities; the discrete
expert counts themselves are not differentiable.

Count-source selection works with both grouping choices and both averaging
choices. It does not change rollout publication, discard behavior, or inference
routing. Matching this setting alone does not establish equivalence with Megatron.

### Configuration examples

The following explicitly spells out the defaults:

```toml
[core]
router_aux_loss_grouping = "pack"
router_aux_loss_reduction = "token"
router_z_loss_reduction = "token"
router_aux_count_source = "dispatch"
router_aux_loss_weight = 0.01
router_z_loss_weight = 1e-5
```

For equal document weighting and current-score balancing counts, change the
following fields in a copied run configuration:

```toml
[core]
compile_model = false
router_aux_loss_grouping = "sequence"
router_aux_loss_reduction = "response"
router_aux_count_source = "current"
# Z-loss stays token-weighted unless router_z_loss_reduction is also changed.
```

Set either coefficient to `0.0` to disable that auxiliary loss; set both to zero
to disable both. These are experimental objective choices, not recommended new
recipe defaults.

### Supported scope and runtime

The all-default pack/token/token objective uses Core's native router methods;
current counts with that objective use Core's native count-source option. Other
grouping/averaging combinations use document metadata preserved across backward
recomputation. They require one unpadded instance per forward (which can contain
multiple packed documents), trainer TP=CP=1, `compile_model=false`, and no global
balancing or router orthogonal loss. Packed and unpacked execution are supported.

Current counts additionally require plain softmax, local balancing, and no EMO
routing, routing biases, expert groups, or uniform/random assignment overrides.
Unsupported native routing combinations fail during model construction. Router
objective controls require the Core MoE backend.

The Core commit pinned in `runtime/miles/runtime.lock.json` includes the router
controls originally introduced at `ab64c3069`.
Build a new application/runtime image from this checkout to use the combined
controls; the earlier image in the GRPO guide does not contain them. Selecting
current counts with an older Core dependency fails explicitly. This source merge
does not promote a new default image.

`open_instruct/test_miles_router_objective.py` checks losses and gradients against
an independent reference for every grouping/averaging/count-source combination,
with and without activation recomputation. It also checks unchanged replay
outputs and policy gradients. The pinned Core source includes native count-source
regressions. `tests/miles/router_objective_contract.py` supplies GPU training and
repacking checks; its `--count-source` option selects either count source. CPU
checks do not establish distributed GPU qualification for every combination.
