# Trainer throughput proposals, 2026-09-14 UTC

Optimize trainer and serving capacity separately on fixed workloads, then size
inference to the measured trainer demand. This document proposes trainer work;
only the already-running packing/concurrency campaign is being exercised now.

## What the current code actually selects

The MoE adapter uses `HFInitializedMoETrainModule`, Core's custom-objective
microbatch/gradient lifecycle, native distributed fused AdamW, and replayed expert
assignments. It already accumulates microbatch gradients before final reduction;
there is no need to add gradient accumulation or replace Adam with a fused optimizer.

- `open_instruct/miles/models.py`: `compile_model=False` is hard-coded.
- `moe_models.py`: native `OLMoDDPOptimizerConfig` is built without `compile`, whose
  Core default is false; DDP config leaves `use_reduce_scatter=False`.
- `moe_models.py`: `recompute_each_block` follows `activation_recompute`, currently true.
- The HF-to-Core factory defaults expert communication to `sync_1d`, capacity 1.25.
- Current qualification arms use 6,144-token packs, FlashAttention 4, dynamic-row
  SwiGLU, forced standalone scoring and exhaustive replay diagnostics.
- The actor already supports guarded scoring skip for one optimizer step per
  collection, zero advantage KL, and a deterministic model. First/periodic checks
  compare scoring and training forwards. Historical behavior probabilities remain
  available to TIS; skipping scoring does not relabel them as current-policy draws.

The audited Core worktree is `miles-core-adapter` at `3d35ab326`. Its hero recipe
inherits model and optimizer compilation from `olmoe3_small_medium_profile.py`;
`olmoe3_small_hero.py::validate` checks 4 x 8,192 tokens per rank microbatch,
no activation checkpointing, no generic float8 config, and reduce-scatter enabled.
The qualified integration bundle has additional explicitly selected kernel flags.
That hero is a different model and runs without EP, so its whole bundle is not an
EP2 RL recommendation.

## Priorities

| Priority | Change | Existing control or implementation? | Expected mechanism and qualification |
|---|---|---|---|
| 1 | Stop forcing the redundant standalone scoring pass | Existing `core.scoring_pass_required=false` and periodic checks | Remove a full no-gradient forward when the recipe admits deriving its old-policy anchor from the unchanged training forward. Check log-probs, TIS, masks, gradients and optimizer updates on the same retained batches. Keep separate reference scoring when required. |
| 1 | Measure production diagnostic level | Existing `core.replay_diagnostics=false` | Leave router replay on, but stop installing/copying/comparing per-layer hooks and synchronizing counters on every pack. Keep exhaustive qualification as a separate run and retain normal contract checks. Quantify the difference before attributing all current overhead to kernels. |
| 1 | Turn off full-block recomputation if memory allows | Existing `trainer.activation_recompute=false` | Avoid repeating the block forward during backward. Test at the established pack size; measure max allocated/reserved memory and headroom. Selective recomputation would need additional plumbing if full off does not fit. |
| 2 | Compile the optimizer | Core capability, needs an explicit wrapper option | The current optimizer is already fused; compilation can further reduce dispatch/intermediate overhead. Profile optimizer time separately first. Start a fresh process and validate optimizer state/resume; Core warns that restored scalar LR versus tensor LR can affect compilation. |
| 2 | Compile model blocks | Core capability, currently disabled in wrapper | Fuse eligible operations and reduce Python/kernel-launch overhead. Start with narrow regions and real variable-length packed inputs. Count graph breaks/recompilations, check document boundaries, routes, gradients and recomputation. Do not assume packing creates static expert shapes. |
| 2 | Evaluate reduce-scatter and bucket policy | Core capability, needs trainer-config plumbing | Reduce applicable gradients into optimizer shards rather than replicating the full reduced gradient. Keep final-microbatch-only synchronization. Expected benefit depends on actual DP groups; EP-owned experts may have little/no replicated-gradient communication on EP2. Validate global norms and updates. |
| 2 | Profile synchronization in the adapter | Implementation work if material | `_agree` performs Gloo object all-gathers; scalar finite checks and per-pack diagnostics force device/host waits. Measure their time and count. Replace expensive reporting/handshakes only while preserving rank-consistent failure handling before collectives. |
| 3 | Evaluate selected pretraining BF16 kernel improvements | Existing guarded Core kernels; RL/EP qualification needed | Pairwise SwiGLU and vectorized FP32 gradient accumulation are candidates. Benchmark the paths actually reached by our model. KDA launch tuning and inverse-scatter selection need the same applicability check. |
| 3 | Compare expert communication implementations | Core has multiple paths; wrapper/qualification work | Current factory chooses synchronous EP. Investigate no-sync or wave overlap only if traces show communication/compute serialization. Replay tensor lifetime, backward buffers and reductions must remain correct. |
| 3 | Avoid materializing all vocabulary logits for RL | New objective/head integration | Core has fused linear cross-entropy machinery, but our custom RL objective requests logits. A chunked/fused selected-token log-probability path could save memory/traffic. It must preserve gradients through the full softmax normalization, entropy/KL options and exact token masks. It is not a switch to ordinary CE. |

Packing itself still has a useful sweep: 6,144 -> 8,192 -> 12,288/16,384 tokens,
holding optimizer batch size and data fixed. This changes tokens per forward,
not the context allowed for a response. Larger packs can improve expert matmuls
and amortize dispatch, but consume activation memory. Compare jointly with the
recomputation decision rather than simply maximizing the pack size.

At fixed trainer GPU count, also compare EP degree versus DP replication if the
model/optimizer fits. This exchanges expert communication and weight residency;
it is not equivalent to adding more GPUs. Keep global batch and normalization fixed.

## Pretraining flags are not a blanket recipe

`olmoe3_integration_policy.py` includes pairwise SwiGLU, vectorized FP32 gradient
accumulation, document-pool routing optimizations, rounded weight gradients,
reduce-scatter shortcuts and optional communication changes. EMO-specific flags
are irrelevant to a model without EMO. Rounded weight-gradient code explicitly
requires a separate EP qualification flag on our topology and changes numerical
behavior. Do not enable the entire bundle by copying its environment variables.
FP8 is a later precision experiment, not the first throughput fix; it needs a
separate numerical, weight-publication and learning qualification.

## Measurement sequence

1. Finish the packed baseline and route/token audits. Report cold and warm phases
   separately; one warm-looking step does not establish a stable plateau.
2. Build a trainer-only replay of retained rollout batches with the same initial
   checkpoint, objective, samples and routing. Use enough distinct batches to
   expose compilation churn. Hold global batch, LR and optimizer-step count fixed.
3. Measure forward/scoring, backward/recomputation, gradient collectives, optimizer,
   validation/CPU waits, peak memory and model/active response tokens/sec/GPU.
   Use a short profiler trace only after warmup. GPU activity is not SM occupancy.
4. First screen: packed baseline versus scoring skip/production diagnostics,
   then recomputation off; choose a pack-size/memory point. Screen optimizer
   compilation and reduce-scatter separately before combining winners.
5. Recheck numerical contracts and resume for any optimizer/reduction change,
   then rerun end-to-end RL to select inference capacity. Keep queue drops,
   blocked producer time, terminal unused work and trainer waits in that report.

## Matched trainer screening campaign

Six EP2 arms replay the same sixteen retained batches from
`steady-2t2i-c32-b128-graphs-986935b17569/run/rollouts`, starting with the same
full-SFT HF weights. Input hashes are recorded per batch. Baseline retains
standalone scoring and exhaustive replay diagnostics. Lean allows the guarded
scoring skip and disables exhaustive replay instrumentation, while retaining
router replay and runtime contracts. Four further arms change one option from
lean: recomputation off, optimizer compilation, model compilation, or native
reduce-scatter. These are experimental switches, not new production defaults.

All arms retain packing at 6144 tokens and `row_specialization="dynamic"`.
The historic runaway specialization was the no-gradient routed SwiGLU buffer
capacity; the qualified dynamic forward from Core commit `307d20590` remains in
place. Backward and wave kernels are unchanged. The compiler observer records
kernel identity, constexpr arguments, cache artifact writes, and Dynamo graph
counts on every batch. In-memory JIT misses include disk-cache hits and must not
be equated with fresh compilation.

Each arm uses an isolated initially cold per-rank cache that persists within the
job. Report startup and cold batches separately from the final warm window;
do not assume six updates suffice without inspecting compiler activity. The
worker measures scoring, forward/loss/backward, optimizer, remaining trainer
bookkeeping, token counts, and CUDA allocator peaks. Synchronization at phase
boundaries aids attribution but may add overhead; use the same instrumentation
for all arms. CUDA allocator peaks are training-phase peaks because the actor
resets peak statistics before training. GPU sampling independently covers the
full job.

This is a trainer throughput screen: no live inference, Ray data transfer,
weight delivery, or learning-curve comparison. Historical behavior probabilities
and routes remain intact. The publication clock advances logically after each
update. Fixed batches become off-policy with respect to each arm's independently
updated parameters; they provide controlled workloads, not a new on-policy RL
experiment. Numerical checks and a live run, including resume qualification for
optimizer changes, are required before promoting a winning option.
