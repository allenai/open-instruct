# Support and remaining parity gaps

MILES + OLMo-core is the preferred GRPO path. This is the current support summary
for the integration; historical evidence retains its original runtime identity. An image qualification
applies to its recorded model, hardware and configuration; a combination of
individually tested flags is not automatically qualified.

| Area | Available and exercised | Limits |
|---|---|---|
| Researcher workflow | Structured TOML, CPU plan/validate, committed-image launch, preparation, receipts/status, native passthrough help | No named olmo-miles recipe catalog; express tasks or adopt a supported immutable manifest |
| MoE training | Earlier KDA/latent SFT models; B300 EP2 learning comparisons, EP1/EP2 numerical tests, larger EP4/EP8 execution | Trainer TP/PP/CP greater than one and microbatch sizes above one are rejected; full SFT MoE colocation and H100 remain unqualified |
| Dense Olmo 3 | Separate Core/FSDP adapter, YaRN reference tests, full 7B two-GPU resident save/resume/export/reload lifecycle | Full-model nonzero-gradient learning and long-run quality still need evidence; see [dense guide](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/olmo3-pre-rl.md) |
| Hybrid MoE (12.5B) | Exact full-checkpoint conversion; paired SFT [four-update barrier checks](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/hero-rl-smoke-20260924.md) with H100 EP4 training, automatic TP1 fused rounding and exact publication | Learning quality, mixed-policy refresh, long context and save/resume remain unqualified; historical HF-reference probability failures are retained |
| Async and replay | Bounded async/TIS, mixed-policy token spans, packing/replay; EP2 and EP8 refresh throughput gates. Earlier barrier-path EP2 fresh-process resume; the completed mixed arms recorded below continued from a native checkpoint under refresh publication | Refresh retains historical behavior probabilities and final-forward replay routes. Continuation there was operator-driven into a fresh output root, and neither it nor the throughput gates establish exact future-sampling reproduction |
| Packing and compute | Document-isolated packing, activation recomputation, dynamic no-gradient SwiGLU rows, checked scoring-pass elision | Default auxiliary loss uses pack-local token averaging; optional [document grouping, averaging and count-source controls](core.md#router-auxiliary-objectives) require the updated runtime |
| Serving capacity | Multiple TP1 engines, independent inference GPU counts, multi-node startup; radix/cache-aware and mixed-chunk exercises | Engine-pool recommendations are workload measurements; TP>1 cache/replay combinations need separate qualification |
| Data and rewards | Math, GSM8K, IF, function and stdio execution, both named judge rubrics; positive natural stdio rewards and mixed groups | Code executor is an external service; judge calibration and the complete published Olmo 3 learning recipe are separate questions |
| Combined workload | Two-node mixed math/IF/code/general runs completed training, evaluation and HF export with refresh publication and a managed judge; see the [archived workload evidence](https://github.com/allenai/open-instruct/blob/7a477910405a65d914b48096f83c13a6c61a60ad/docs/miles/feature-parity.md) | Endurance and lifecycle evidence only. Held-out movement was weak and mixed; this is not a learning or throughput qualification, and the earlier transport failure remains unexplained |
| Checkpoints | Synchronous native saves, topology/cursor validation, same-topology restore, final HF export and fresh dense serving reload | Rolling retention is implemented; background saves and token-per-expert cadence are not ported. [Two-replica restart](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/multinode-resume-20260922.md) resumes from the newest checkpoint and continues the rollout cursor; forced multi-node preemption, and restarts of runs carrying a managed judge, remain unqualified |
| Health and recovery | Separate health/generation connections, stale-probe guards, bounded HTTP retries and explicit failures | No general automatic trainer recovery or qualified engine-replacement/republish lifecycle; restore a completed checkpoint |
| Publication | Streamed/fused tensors, flattened NCCL or colocated IPC, startup full-weight audit; periodic full audits default off | Current full-model starters select mixed-policy refresh. Barrier remains the low-level default; [engine drain](engine-drain.md) finishes requests before swapping and is a separate mode |
| Compiler caches | Private local Triton caches, verified immutable shared generations, bounded best-effort publication | Other compiler families and CUDA graphs are not persisted by the Ray integration |
| Background evaluation | Opt-in `evaluation.mode="background"`: independent Beaker evaluator jobs that never borrow, drain or pause rollout engines; [tiny MoE mechanics qualification](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/background-evaluation-20260920.md) with accepted W&B status semantics | Best effort by design: a busy submitter drops milestones, failures are not retried and preemption can interrupt submission. The large template is provisional, and architecture, task, tensor-parallel size and evaluator image each need their own qualification |
| Long sequences | Actual 16K/32K/64K input serving probes; bounded 16K and 32K RL exercises | Read [length evidence](long-sequences.md): serving success does not establish 64K backward or high-concurrency memory fit |

Build from the selected application revision and [runtime lock](architecture.md#runtime-sources-and-images). Historical image results below apply only to their recorded sources and configurations.

## Evidence and validation

The [archived evidence index](https://github.com/allenai/open-instruct/blob/7a477910405a65d914b48096f83c13a6c61a60ad/docs/miles/feature-parity.md#evidence-to-read-first)
links the filtering, packing, replay, mixed-workload and lifecycle measurements.
Keep lifecycle completion, nonzero learning signal and specific-feature coverage
as separate conclusions. For current commands, follow
[architecture and development](architecture.md#local-development).

## Differences from olmo-miles to retain explicitly

Core owns its optimizer/checkpoint lifecycle and parallelism semantics; a native
Megatron flag is not a substitute. See [run controls](run-controls.md) for exact
mappings and rejected settings, and the generated [configuration reference](configuration.md)
for the accepted surface. Trainer offload, async checkpoint writing, token-per-expert checkpoint
cadence, the full recipe catalog, some external judge modes, and advanced
parallel layouts remain implementation gaps. Selecting larger batch sizes or more
engines cannot supply those features.
