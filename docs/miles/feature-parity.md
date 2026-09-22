# Support and remaining parity gaps

MILES + OLMo-core is the preferred GRPO path. This is the current support summary
for the project branches, updated September 21, 2026; historical evidence retains its original runtime identity. An image qualification
applies to its recorded model, hardware and configuration; a combination of
individually tested flags is not automatically qualified.

| Area | Available and exercised | Boundary for colleagues |
|---|---|---|
| Researcher workflow | Structured TOML, CPU plan/validate, committed-image launch, preparation, receipts/status, native passthrough help | No named olmo-miles recipe catalog; express tasks or adopt a supported immutable manifest |
| MoE training | Earlier KDA/latent SFT models; B300 EP2 learning comparisons, EP1/EP2 numerical tests, larger EP4/EP8 execution | Trainer TP/PP/CP greater than one and microbatch sizes above one are rejected; full SFT MoE colocation and H100 remain unqualified |
| Dense Olmo 3 | Separate Core/FSDP adapter, YaRN reference tests, full 7B two-GPU resident save/resume/export/reload lifecycle | Full-model nonzero-gradient learning and long-run quality still need evidence; see [dense guide](olmo3-pre-rl.md) |
| Hero | Config-driven conversion and exact full-checkpoint tensor round trip, tiny attention/gain/scale tests | The full-checkpoint probability gate failed; hero RL is not an accepted sharing baseline |
| Async and replay | Bounded async/TIS, mixed-policy token spans, packing/replay; EP2 and EP8 refresh throughput gates. Earlier barrier-path EP2 fresh-process resume; the completed mixed arms recorded below continued from a native checkpoint under refresh publication | Refresh retains historical behavior probabilities and final-forward replay routes. Continuation there was operator-driven into a fresh output root, and neither it nor the throughput gates establish exact future-sampling reproduction |
| Packing and compute | Document-isolated packing, activation recomputation, dynamic no-gradient SwiGLU rows, checked scoring-pass elision | Default auxiliary loss uses pack-local token averaging; optional [document grouping, averaging and count-source controls](core.md#router-auxiliary-objectives) require the updated runtime |
| Serving capacity | Multiple TP1 engines, independent inference GPU counts, multi-node startup; radix/cache-aware and mixed-chunk exercises | Engine-pool recommendations are workload measurements; TP>1 cache/replay combinations need separate qualification |
| Data and rewards | Math, GSM8K, IF, function and stdio execution, both named judge rubrics; positive natural stdio rewards and mixed groups | Code executor is an external service; judge calibration and the complete published Olmo 3 learning recipe are separate questions |
| Combined workload | Three two-node arms each completed 100 updates of the four-domain mixed workload and exited zero after final evaluation and HF export: [DP4/EP2 control](https://beaker.org/ex/01M2ZVH6TQVS037NPD2CWKWXBH), [DP4/EP2 treatment](https://beaker.org/ex/01M308PE9GNF05CHHETTCBZMRP) and [EP8 reference](https://beaker.org/ex/01M2ZVGQ4EB18YN2KMQPBE54XX), each with seven TP1 engines and one managed judge, refresh publication and a 32K response cap. The recovered segment's audit passed: 60 optimizer steps, 24 scoring checks, maximum scoring difference zero, eight ranks, finite, behaviour age respected | Endurance and lifecycle only. Held-out movement was weak and mixed, so this is not a learning or throughput qualification, and the earlier single-update EP8 transport failure is superseded rather than explained |
| Checkpoints | Synchronous native saves, topology/cursor validation, same-topology restore, final HF export and fresh dense serving reload | Rolling retention is implemented; background saves and token-per-expert cadence are not ported. [Two-replica restart](measurements/multinode-resume-20260922.md) resumes from the newest checkpoint and continues the rollout cursor; forced multi-node preemption, and restarts of runs carrying a managed judge, remain unqualified |
| Health and recovery | Separate health/generation connections, stale-probe guards, bounded HTTP retries and explicit failures | No general automatic trainer recovery or qualified engine-replacement/republish lifecycle; restore a completed checkpoint |
| Publication | Streamed/fused tensors, flattened NCCL or colocated IPC, startup full-weight audit; periodic full audits default off | Current full-model starters select mixed-policy refresh. Barrier remains the low-level default; [engine drain](engine-drain.md) finishes requests before swapping and is a separate mode |
| Compiler caches | Private local Triton caches, verified immutable shared generations, bounded best-effort publication | Other compiler families and CUDA graphs are not persisted by the Ray integration |
| Background evaluation | Opt-in `evaluation.mode="background"`: independent Beaker evaluator jobs that never borrow, drain or pause rollout engines; [tiny MoE mechanics qualification](measurements/background-evaluation-20260920.md) with accepted W&B status semantics | Best effort by design: a busy submitter drops milestones, failures are not retried and preemption can interrupt submission. The large template is provisional, and architecture, task, tensor-parallel size and evaluator image each need their own qualification |
| Long sequences | Actual 16K/32K/64K input serving probes; bounded 16K and 32K RL exercises | Read [length evidence](long-sequences.md): serving success does not establish 64K backward or high-concurrency memory fit |

Use image **`01M2XGZM2N1V4DQVMYHM52KBHZ`**; [MILES GRPO](grpo.md) records its
source identity, publication modes and qualification boundaries.

## Evidence to read first

- [Online filtering with small barrier and refresh runs](measurements/online-filtering-20260919.md):
  all-zero/all-one groups dropped, accepted batches replenished, four updates each;
  synthetic rewards establish mechanics, not learning quality.

- [Current router controls and standard examples](measurements/router-controls-20260918.md):
  dev/small lifecycle and retention passed; medium and optional controls remain
  in qualification. Large is excluded at the user’s request.

- [Merged refresh and fast EP2 qualification](measurements/full-sft-basket-20260914.md),
  [EP8 GSM8K throughput](measurements/throughput-20260913.md) and
  [current profiles](throughput-profiles.md).
- [Learning comparisons](measurements/gsm8k-results-20260911.md) and
  [light-SFT results](measurements/light-sft1000-gsm8k.md): task behavior and
  checkpoint-specific comparisons, not a guarantee of identical trainers.
- [Readiness continuation](measurements/colleague-20260913/README.md): dense and
  MoE restart, code/judges, cached-prefix replay, combined workload and failure scope.
- [Packing](measurements/sequence-packing-20260912/README.md),
  [full replay](measurements/core-replay-full-sft-20260911.md), and
  [length guidance](measurements/length-guidance-20260913/README.md): concrete gates.
- [Engine drain](measurements/engine-drain-20260913/README.md): independent drain,
  resume, immutable transfer and bounded terminal-failure evidence.

The implementation is suitable for a bounded colleague pilot on these exercised
paths. Long baselines on the same recorded runtime image provide endurance and learning
evidence; they do not require inventing a broader failure-recovery system first.
Keep lifecycle completion, nonzero learning signal and specific-feature coverage
as separate conclusions in each report.

## Differences from olmo-miles to retain explicitly

Core owns its optimizer/checkpoint lifecycle and parallelism semantics; a native
Megatron flag is not a substitute. See [run controls](run-controls.md) for exact
mappings and rejected settings, and the generated [configuration reference](configuration.md)
for the accepted surface. Trainer offload, async checkpoint writing, token-per-expert checkpoint
cadence, the full recipe catalog, some external judge modes, and advanced
parallel layouts remain implementation gaps. Selecting larger batch sizes or more
engines cannot supply those features.

The earlier [parity snapshot](measurements/implementation-history/feature-parity-before-sharing-20260913.md)
and September 13 consolidation list are historical. Mixed-policy refresh and
throughput profiles are now merged into the primary Open Instruct/MILES branches.
The recorded code-service/producer transport failure and unqualified combinations
remain visible; merge status is separate from the breadth of runtime evidence.
