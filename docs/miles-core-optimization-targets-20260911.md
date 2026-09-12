# Core RL path: optimization targets

Written 2026-09-11 from the completed 500-update Core and Megatron GSM8K runs
(`.artifacts/miles-learning-watch-20260911/`). Each target below is meant to be
planned and executed independently, one at a time, with its own acceptance check.
Numbers are from native log timestamps; the script that produced the phase
breakdown is `.artifacts/miles-learning-watch-20260911/analysis/phase_breakdown.py`.

Ordering is by expected saving per 500-update run. Items marked **shared** change
both arms and must land in both to keep comparisons paired. Items marked **Core**
touch only the Core trainer or adapter.

## Baseline

The run is a synchronous loop: generate, score, train, publish, repeat. Three B300
GPUs: two trainer ranks (expert parallel 2), one SGLang serving GPU. Either the
serving GPU or the trainer GPUs are working at any moment, never both.

### Where Core's 10.4 hours went

| Bucket | Hours | Share |
|---|---:|---:|
| Warm cycles, updates 0-199 (compilation-inflated) | 4.4 | 42% |
| Warm cycles, updates 200-499 | 3.2 | 31% |
| Checkpoint saves, 5 x ~1000 s | 1.4 | 13% |
| Evaluations, 25 x ~170 s | 1.2 | 11% |
| Startup to first update | 0.24 | 2% |

### Steady-state warm cycle, updates 300-499

| Phase | Core | Core % | Megatron | Megatron % |
|---|---:|---:|---:|---:|
| Ingress + scoring (old log-probs) | 8.7 s | 22 | 5.0 s | 9 |
| Training (forward, backward, optimizer) | 5.4 s | 14 | 22.4 s | 39 |
| Debug dump before publish | - | - | 1.3 s | 2 |
| Publication (weights to SGLang) | 3.8 s | 10 | 6.6 s | 12 |
| Orchestration gaps | 0.01 s | 0 | 0.04 s | 0 |
| Generation (SGLang, 16 samples) | 20.7 s | 54 | 21.5 s | 38 |
| **Cycle** | **38.6 s** | | **56.8 s** | |

### Warm cycle by window

| Updates | Core cycle | Core scoring | Core train | Megatron cycle | Megatron scoring | Megatron train |
|---|---:|---:|---:|---:|---:|---:|
| 0-99 | 93.7 | 46.5 | 6.2 | 69.0 | 4.5 | 19.2 |
| 100-199 | 63.2 | 27.3 | 5.5 | 63.5 | 4.5 | 20.1 |
| 200-299 | 37.1 | 11.9 | 5.4 | 53.9 | 4.4 | 20.1 |
| 300-399 | 35.3 | 8.0 | 5.2 | 46.8 | 3.7 | 16.1 |
| 400-499 | 41.9 | 9.3 | 5.5 | 66.8 | 6.3 | 28.6 |

Facts established along the way that the targets depend on:

- Generation is the same SGLang engine in both arms. Its time tracks total batch
  tokens (correlation 0.98), so it is throughput-bound at 4 concurrent requests,
  not latency-bound by the longest response.
- Core scoring time tracks batch tokens (0.87) while Core training does not (0.03).
  Training runs a forward, a recomputation forward, and a backward in a flat 5.4 s.
  Scoring's token dependence is therefore compilation, not compute. Identical
  repeat batches score in 1.1-1.2 s.
- Greedy evaluation is not repeatable across launches without pinning Triton
  autotune choices; with the seven FLA tuner choices pinned, two launches match
  bit for bit. Evaluation differences of a few questions are noise.
- GRPO here is exactly one optimizer step per rollout, so the PPO ratio is 1 by
  construction and the batch loss is identically zero; only gradients carry signal.

## Measurement protocol for every target

- Run the phase breakdown script on the new run's `final.log`. Phases must still
  sum to the cycle.
- Report steady state (updates 300-499) and the 0-99 window separately.
- For any change that touches numerics, compare log-probabilities on retained
  batches against the parent bit for bit, as the scorer qualification did.
- For cross-process comparisons, pin FLA autotune configurations first or the
  comparison measures kernel-selection noise.

---

## 1. Scoring recompilation (Core) - about 2.3 h per run

**Mechanism.** OLMo-core's routed experts use a fused Triton SwiGLU kernel only
under `no_grad` (`src/olmo_core/nn/moe/v2/routed_experts.py:1119`). The kernel
declares the routed row capacity as `tl.constexpr`, so every new capacity compiles
a new binary. Capacity changes with every batch. Measured: about 150 new variants
per rank per batch, 41-43 s of JIT per batch. The gradient path uses eager SwiGLU
and never recompiles, which is why training time is flat.

**Change.** Runtime-row kernel: `rows` becomes a plain argument marked
`do_not_specialize`, launch-grid stride computed at runtime. Patch:
`scripts/miles/diagnostics/swiglu-runtime-rows.patch`. Qualified twice, bit-exact
on 154,531 log-probabilities; variants go from ~150 per batch to one total;
changing-batch scoring 60.3 s to 16.2 s. Land behind a wrapper argument
`row_specialization` (static or dynamic) resolved once from Core config at trainer
initialization, not a runtime scope. Apply the same change to the backward kernel
for consistency even though the gradient path does not use it today.

**Also.** Persistent Triton cache directory on WEKA (`TRITON_CACHE_DIR`). FLA
already passes `cache_results=True` to Triton, but each Beaker allocation starts
cold, so every launch re-tunes and re-compiles. A persistent cache removes the
cold-start cost (first-100-update scoring at 46 s per cycle) and the residual FLA
misses that remain after the row fix. It also pins autotune choices across
launches, which is the determinism fix for evaluations.

**Acceptance.** Scoring time no longer correlates with batch tokens; window 0-99
scoring within a few seconds of window 300-399; per-rank SwiGLU variant count is
one; log-probabilities bit-identical to parent on retained batches.

**Risk.** Low. Arithmetic unchanged. Audit other Core Triton kernels for
data-dependent `constexpr` arguments while there.

## 2. Checkpoint save path (Core) - up to 1 h per run

**Mechanism.** `save_state_dict_direct` in
`src/olmo_core/train/train_module/transformer/ddp_train_module.py:1119`: build the
optimizer state dict (cheap, views), write with `RemoteFileSystemWriter`
(`src/olmo_core/distributed/checkpoint/filesystem.py`), write metadata, reload the
optimizer state. The writer splits each rank's ~120 GB into thread buckets and for
every tensor does a GPU-to-CPU copy, a `.clone()`, and a per-tensor `torch.save`
into a temp file, under the interpreter lock. Replicated tensors are deduplicated
onto rank 0, so rank 1 idles at the barrier. Observed: about 1000 s per save
(0.24 GB/s aggregate) versus Megatron's ~340 s for the same state. No timers
exist inside the Core save, so the split between write and reload is not measured.

**Change, in order.**
1. Add timers around state-dict construction, write, metadata, and reload, logged
   like publication is.
2. Use the writer's existing `process_count` path instead of threads; plumb it
   through `save_state_dict_direct`.
3. Replace the unconditional `.clone()` with a contiguity check.
4. Distribute replicated tensors across ranks rather than assigning all to rank 0.
5. Asynchronous save overlapping the next rollouts (after the write is fast).
6. Reconsider cadence: resumable save every 200, model-only export in between.

**Acceptance.** Save time at or below Megatron's 340 s; resume from the new
checkpoint reproduces the next optimizer step's log-probabilities exactly.

**Risk.** Medium. Touches the resume contract; the existing checkpoint topology
tests must pass and a resume must be exercised.

## 3. Evaluation concurrency (shared) - about 0.8 h per run

**Mechanism.** 128 greedy prompts at 4 concurrent requests take about 170 s, 25
times per run. At 32 or more concurrent the phase is bounded by the longest
response, roughly 40-60 s.

**Change.** Raise `sglang_server_concurrency` and `sglang_max_running_requests`
for evaluation. Raise `sglang_max_mamba_cache_size` (KDA layers hold per-request
state) and `sglang_cuda_graph_max_bs_decode` alongside. Same values in both arms.

**Acceptance.** Evaluation wall time under 60 s; scores within noise of previous
runs (they are noise-bound regardless of concurrency).

**Risk.** Low. Memory: check the static memory fraction still leaves room for the
larger KV and mamba caches.

## 4. Rollout concurrency and group size (shared) - about 1 h per run, more with larger groups

**Mechanism.** Generation is 54% of Core's steady cycle and throughput-bound at 4
concurrent with 16 sequences per update. All 16 in flight would cut it an
estimated 30-40%. Separately, only ~30% of groups had mixed rewards at group size
4, so ~70% of generated data produced no policy gradient.

**Change.** Concurrency 16 for training rollouts (same cache and graph caps as
item 3). Then consider larger groups: sample 4 per prompt, then 12 more only for
prompts whose first 4 were not all correct. The MILES reward normalizer already
accepts per-prompt group sizes; the data source currently asserts a fixed count.
Note the loss weighting consequence: per-response mean then sum, so a 16-sample
prompt weighs four times a 4-sample prompt; decide whether that is wanted or
normalize per prompt.

**Acceptance.** Generation time per update drops and becomes bounded by the
longest response; fraction of mixed-reward groups rises.

**Risk.** Low for concurrency; medium for variable groups (recipe change, needs
both arms and a learning comparison).

## 5. Skip the standalone scoring pass (Core first, Megatron identical) - 2-3 s per update

**Mechanism.** With one optimizer step per rollout, "old" and "new"
log-probabilities are computed at the same weights, so the training forward's own
detached log-probabilities are exactly the old values. MILES supports this via
`skip_actor_forward_only`. The Core adapter calls `_score` unconditionally
(`open_instruct/miles/actor.py:130`).

**Change.** Config flag defaulting to auto. Auto skips only when all hold: global
batch equals rollout size (one step per rollout), synchronous rollouts, KL
coefficient zero, not using rollout log-probabilities as old. Otherwise run the
full pass. Assert loudly on any other configuration. Add a periodic diagnostic
scoring pass (every N updates) that reports scoring-versus-training discrepancy,
since that comparison is what found the SwiGLU rounding bug. Move the SGLang
agreement check to read from the training forward; it runs inside the gradient
step across ranks, so place its collective where every rank reaches it.

**Acceptance.** Gradients and optimizer updates identical to the parent on a
retained batch; periodic diagnostic reports zero discrepancy after item 1.

**Risk.** Low arithmetic, medium engineering. Sequence after item 1 so each run
changes one variable.

## 6. Publication transport (Core) - about 0.35 h per run

**Mechanism.** 3.6 s per publication, of which 3.3 s is transporting 37 GB in 35
sequential 1 GiB buckets (about 11 GB/s). Export pack is 0.25-0.45 s, finalize
0.05 s. Megatron spends 3.4 s of its 6.6 s waiting for the engine to load.

**Change.** Overlap pack and send with buckets in flight concurrently, or larger
buckets. Confirm what the engine-side load costs; the transport figure may hide
engine load time.

**Acceptance.** Publication under 1.5 s; full serving-weight equality check
still passes.

**Risk.** Low-medium. Weight-sync correctness is checked by the existing full
equality check; keep it on for the qualifying run.

## 7. Training step fixed overhead (Core) - perhaps 0.3 h per run

**Mechanism.** 5.4 s regardless of batch tokens (10k to 19k). Compute is
therefore under about 2 s and the remainder is fixed: optimizer step over ~9B
parameters per rank with FP32 master weights, gradient reduction of replicated
parameters, activation recomputation setup, and the contract checks, several of
which perform collectives via `_agree`.

**Change.** Profile one step before touching anything. Candidates: fuse or batch
the optimizer update; move diagnostic checks off the critical path or behind
`diagnostic_interval`.

**Acceptance.** Training time falls and begins to scale with tokens, which means
the fixed part is gone.

**Risk.** Low until a change is chosen.

## 8. Asynchronous overlap of generation and training (shared, design change)

**Mechanism.** Synchronous cycle is the sum of the two sides. Asynchronous is
roughly the longer of the two. After items 1-7, Core's trainer side is about 8 s
against 13-20 s of generation, so the cycle becomes generation-bound and the
trainer GPUs idle most of the time. At that point the GPU ratio should shift
toward serving.

**Change.** MILES asynchronous rollout mode with truncated importance sampling
using SGLang log-probabilities as "old". The Core adapter currently rejects
retries, witnesses, and external data; it needs async support. Async trial
scripts and a scheduling note exist in the worktree.

**Acceptance.** Cycle approaches max(generation, trainer); learning curve within
noise of synchronous on the same recipe. Do not combine with any other change in
the qualifying run.

**Risk.** High. Changes the algorithm (staleness correction becomes live). Last.

## 9. Startup (Core) - about 10 min per run

**Mechanism.** 14.4 min from process start to first update. Earlier measurement:
process start to first publication 653 s, dominated by HF import into native
topology and cold compilation.

**Change.** Persistent Triton cache (item 1) covers the compilation share. A
native checkpoint load instead of HF import for the initial weights covers the
rest, but only matters once runs are short enough for startup to be a
noticeable fraction.

**Risk.** Low. Lowest priority.

---

## Expected outcome

Items 1-4 alone take the 500-update Core run from about 10.4 h to roughly 5.5 h
with no change to the learning algorithm. Items 5-7 take another 15-20 min off
and simplify the contract. Item 8 roughly halves what remains but is a recipe
change and should be qualified on its own.
