# Support and remaining parity gaps

MILES + OLMo-core is the preferred GRPO path. This is the current support summary
for the project branches, reviewed September 13, 2026. An image qualification
applies to its recorded model, hardware and configuration; a combination of
individually tested flags is not automatically qualified.

| Area | Available and exercised | Boundary for colleagues |
|---|---|---|
| Researcher workflow | Structured TOML, CPU plan/validate, committed-image launch, preparation, receipts/status, native passthrough help | No named olmo-miles recipe catalog; express tasks or adopt a supported immutable manifest |
| MoE training | Earlier KDA/latent SFT models; B300 EP2 learning comparisons, EP1/EP2 numerical tests, larger EP4/EP8 execution | Trainer TP/PP/CP greater than one and microbatch sizes above one are rejected; full SFT MoE colocation and H100 remain unqualified |
| Dense Olmo 3 | Separate Core/FSDP adapter, YaRN reference tests, full 7B two-GPU resident save/resume/export/reload lifecycle | Full-model nonzero-gradient learning and long-run quality still need evidence; see [dense guide](olmo3-pre-rl.md) |
| Hero | Config-driven conversion and exact full-checkpoint tensor round trip, tiny attention/gain/scale tests | The full-checkpoint probability gate failed; hero RL is not an accepted sharing baseline |
| Async and replay | Bounded async, TIS, homogeneous versioned groups, packing/replay/recomputation; full-model EP2 fresh-process resume | Same trainer topology on resume; no exact reproduction of future sampled rollouts |
| Packing and compute | Document-isolated packing, activation recomputation, dynamic no-gradient SwiGLU rows, checked scoring-pass elision | Auxiliary loss uses Core local-batch tokens within each packed forward; packing changes that objective relative to separate samples |
| Serving capacity | Multiple TP1 engines, independent inference GPU counts, multi-node startup; radix/cache-aware and mixed-chunk exercises | Engine-pool recommendations are workload measurements; TP>1 cache/replay combinations need separate qualification |
| Data and rewards | Math, GSM8K, IF, function and stdio execution, both named judge rubrics; positive natural stdio rewards and mixed groups | Code executor is an external service; judge calibration and the complete published Olmo 3 learning recipe are separate questions |
| Combined workload | Four-update EP2 + three policy engines + one judge: async/TIS, packing/replay, actual radix hits, saves, six reward domains | Full EP8 + seven engines + judge combination and sustained throughput remain separate gates |
| Checkpoints | Synchronous native saves, topology/cursor validation, same-topology restore, final HF export and fresh dense serving reload | Background saves, retention policy and token-per-expert cadence are not ported; multi-node/managed automatic restart is disabled |
| Health and recovery | Separate health/generation connections, stale-probe guards, bounded HTTP retries and explicit failures | No general automatic trainer recovery or qualified engine-replacement/republish lifecycle; restore a completed checkpoint |
| Publication | Streamed/fused tensors, flattened NCCL or colocated IPC, startup full-weight audit; periodic full audits default off | Barrier publication is the normal path; [engine drain](engine-drain.md) is an opt-in qualified experiment, not mixed-policy decoding |
| Compiler caches | Private local Triton caches, verified immutable shared generations, bounded best-effort publication | Other compiler families and CUDA graphs are not persisted by the Ray integration |
| Long sequences | Actual 16K/32K/64K input serving probes; bounded 16K and 32K RL exercises | Read [length evidence](long-sequences.md): serving success does not establish 64K backward or high-concurrency memory fit |

## Evidence to read first

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
paths. Long baselines on a single candidate image provide endurance and learning
evidence; they do not require inventing a broader failure-recovery system first.
Keep lifecycle completion, nonzero learning signal and specific-feature coverage
as separate conclusions in each report.

## Differences from olmo-miles to retain explicitly

Core owns its optimizer/checkpoint lifecycle and parallelism semantics; a native
Megatron flag is not a substitute. See [run controls](run-controls.md) for exact
mappings and rejected settings, and the generated [configuration reference](configuration.md)
for the accepted surface. Trainer offload, async checkpoint writing, checkpoint
retention/cadence, the full recipe catalog, some external judge modes, and advanced
parallel layouts remain implementation gaps. Selecting larger batch sizes or more
engines cannot supply those features.

The earlier [parity snapshot](measurements/implementation-history/feature-parity-before-sharing-20260913.md)
is retained as development history. Its old pending-work list is superseded by
this page. Mixed-policy refresh and throughput-profile work on separate feature
branches is not part of the sharing candidate unless explicitly promoted after
its full training gate.
