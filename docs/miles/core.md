# MILES with an OLMo-core trainer

Use the [operating guide](index.md) for new GRPO runs and the
[support matrix](feature-parity.md) for qualified boundaries. The adapter replaces
the MILES Megatron trainer actor, while retaining MILES rollout actors, samples,
advantages, PPO loss helpers and transport primitives. Open Instruct supplies
configuration, data/verifiers, lifecycle coordination and retained evidence.

## Trainer hookup

`open_instruct/miles/actor.py` implements the trainer actor contract. The
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

The original Core adapter used Jacob's `jacobm/moe-v2-core-gdn2` branch. Its current
base is `codex/small-hero-hf-20260909` revision
`b1fd2c9746e88baeb20e372bdca340d788d0f7e5`, with checksum-verified adapter patches.
That lineage retains the earlier KDA/latent model. Use the
[runtime lock and build procedure](architecture.md#runtime-sources-and-images),
not a sibling checkout or an old branch name, to reproduce the runtime.

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
