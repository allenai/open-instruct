# Support and limits

MILES + OLMo-core is the GRPO path for new work. This page summarizes what the
integration supports and its current limits. Support for one model, topology or
feature does not imply support for every combination; check a new combination
on a small run first. For starting settings see [development defaults](development-defaults.md).

| Area | Available | Limits |
|---|---|---|
| Researcher workflow | Structured TOML, CPU plan/validate, committed-image launch, preparation, receipts/status, native passthrough help | No named recipe catalog; express tasks or adopt an immutable manifest |
| MoE training | Core MoE adapter with expert parallelism, router replay and router auxiliary-loss controls | Trainer TP/PP/CP greater than one and microbatch sizes above one are rejected; full-model colocation is not supported |
| Dense Olmo 3 | Separate Core/FSDP adapter with save, resume, HF export and serving reload | Learning behavior is less explored than the MoE path |
| Hybrid MoE | Checkpoint conversion, EP training and TP1 serving with [Core-compatible rounding](core-compatible-serving.md) | See [hybrid MoE support](moe-model-support.md) |
| Async and replay | Bounded policy lag with TIS, mixed-policy refresh, packing with replay | Refresh keeps historical behavior probabilities; a resumed run does not reproduce the exact future samples |
| Packing and compute | Document-isolated packing, activation recomputation, dynamic SwiGLU rows, guarded scoring-pass skip | Auxiliary loss defaults to pack-local token averaging; see [router auxiliary objectives](core.md#router-auxiliary-objectives) |
| Serving | Multiple TP1 engines, independent inference GPU counts, multi-node startup, radix and cache-aware routing | TP>1 engines with cache/replay combinations are untested |
| Data and rewards | Math, GSM8K, IF, function and stdio code execution, named judge rubrics | The code executor is an external service |
| Checkpoints | Synchronous native saves with rolling retention, topology/cursor validation, same-topology restore, final HF export, multi-node restart from the newest checkpoint | Background saves, topology changes on restore and restarts of runs with a managed judge are not supported or untested |
| Health and recovery | Separate health/generation connections, bounded HTTP retries, whole-group retry of failed generations | No automatic trainer recovery or engine replacement; restart from a completed checkpoint |
| Publication | Streamed fused tensors over NCCL or colocated IPC, startup weight audit | Barrier is the low-level default; [engine drain](engine-drain.md) is experimental |
| Compiler caches | Private local Triton caches with shared immutable generations and bounded best-effort publication | Other compiler families and CUDA graphs are not persisted |
| Background evaluation | Opt-in independent Beaker evaluator jobs that never pause rollout engines | Best effort: milestones can be skipped, and failed submissions are not retried |
| Long sequences | Prompt/response budgets and serving controls for 16K–64K contexts | Trainer memory at long context depends on the model and topology; see [long sequences](long-sequences.md) |

## Differences from the Megatron implementation

Core owns its optimizer/checkpoint lifecycle and parallelism semantics; a native
Megatron flag is not a substitute. See [run controls](run-controls.md) for
mappings and rejected settings, and the generated [configuration reference](configuration.md)
for the accepted surface. Trainer offload, async checkpoint writing, token-per-expert
checkpoint cadence, a named recipe catalog, some external judge modes and advanced
parallel layouts are not implemented.
