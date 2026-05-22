# OLMo-core in open-instruct: a reference

This page is a map of **what open-instruct uses [OLMo-core](https://github.com/allenai/OLMo-core) for**, where each piece is wired in, and how those choices compare to the alternatives we could have picked (HuggingFace Transformers + DeepSpeed / Accelerate, vanilla PyTorch FSDP, torchtitan, Megatron-LM).

The companion page [OLMo-core Sharding and Parallelism](olmo_core_sharding.md) drills into HSDP/FSDP2 configuration. This page is the higher-level "where does it touch the codebase, and why this framework".

If terms like FSDP, ZeRO, DeepSpeed, Accelerate, Ray, or mixed precision are unfamiliar, read the [Distributed Training Primer](../distributed_training_primer.md) first — it covers the concepts this page assumes.

## What OLMo-core is

OLMo-core is Ai2's training framework for the OLMo model family. It is the modern successor to the original `OLMo` repo and is *not* a model zoo on top of HuggingFace — it is a self-contained training stack:

- Custom `Transformer` / attention / MoE / LM-head modules tuned for FSDP2 + `torch.compile`.
- A DTensor-based parallelism layer (FSDP2, HSDP, TP, eventually CP).
- A `Trainer` + `TrainModule` + `Callback` system (closer to torchtitan / pytorch-lightning style than HF Trainer).
- HF-format checkpoint load/save bridges, custom CUDA kernels, float8, fused AdamW + Muon optimizers.

Open-instruct pulls it in as a pinned git dependency:

```toml
# pyproject.toml
ai2-olmo-core = { git = "https://github.com/allenai/OLMo-core.git", rev = "002958a8f15a" }
```

## Where OLMo-core is used in this repo

OLMo-core is the **primary training backend for DPO and the new GRPO path**. SFT delegates to OLMo-core's own SFT script. The older "fast" GRPO path (`grpo_fast.py`) and the deprecated finetune path still use HF + DeepSpeed / Accelerate.

| Training path | Framework | Entry point | Distributed |
|---|---|---|---|
| DPO | **OLMo-core** | [open_instruct/dpo.py](../../open_instruct/dpo.py) | `torch.distributed` + FSDP2/HSDP (+ optional TP) |
| GRPO (new) | **OLMo-core** | [open_instruct/grpo.py](../../open_instruct/grpo.py) → [grpo_olmo_core_actor.py](../../open_instruct/grpo_olmo_core_actor.py) | Ray actors, each running FSDP2/HSDP |
| GRPO-Fast | HF + DeepSpeed | [open_instruct/grpo_fast.py](../../open_instruct/grpo_fast.py) | DeepSpeed ZeRO-3 |
| SFT (recommended) | OLMo-core (external) | [OLMo-core/src/scripts/train/sft](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft) | OLMo-core stack |
| SFT (deprecated) | HF + Accelerate | [open_instruct/finetune.py](../../open_instruct/finetune.py) | DeepSpeed via Accelerate |

### The OLMo-core wrapper layer

The OLMo-core integration is concentrated in five files. Together they are the boundary between "open-instruct config / data" and "OLMo-core Trainer":

| File | What it owns |
|---|---|
| [open_instruct/olmo_core_utils.py](../../open_instruct/olmo_core_utils.py) | Shared config dataclasses (`ExperimentConfig`, `ModelConfig`, `TrainingConfig`, `DatasetConfig`, `LoggingConfig`, `CheckpointConfig`), the `OLMO_MODEL_CONFIG_MAP` (HF name → OLMo-core `TransformerConfig` name), `setup_model`, `save_state_dict_as_hf`, distributed dataset loading, AC config builder, checkpoint callback factory. |
| [open_instruct/olmo_core_train_modules.py](../../open_instruct/olmo_core_train_modules.py) | `DPOTrainModule` and `GRPOTrainModule` — subclasses of OLMo-core's `TransformerTrainModule` that implement the DPO / GRPO loss, log-prob computation, and DTensor-aware forward paths. Also defines `DPOLMHead`, a torch.compile-friendly LM head that returns per-token log-probs. |
| [open_instruct/olmo_core_callbacks.py](../../open_instruct/olmo_core_callbacks.py) | `BeakerCallbackV2` (Beaker v2 description / metadata updates) and `PerfCallback` (MFU + tokens/sec). |
| [open_instruct/grpo_callbacks.py](../../open_instruct/grpo_callbacks.py) | GRPO-only callbacks: `RefPolicyUpdateCallback`, `VLLMWeightSyncCallback`, and `olmo_core_to_hf_name` (param-name translation when streaming weights to vLLM). |
| [open_instruct/grpo_olmo_core_actor.py](../../open_instruct/grpo_olmo_core_actor.py) | `PolicyTrainerOLMoCoreProcess` — the Ray actor that owns one shard of the policy and drives the OLMo-core `Trainer` from inside Ray. |

### What OLMo-core APIs we actually consume

A tour of the OLMo-core surface area that open-instruct depends on:

- **Model definitions** — `olmo_core.nn.transformer.{Transformer, TransformerConfig}`. We never instantiate HF model classes for OLMo-core-backed training; we build the OLMo-core `Transformer` on CPU and then load HF weights into the sharded model.
- **Attention backends** — `olmo_core.nn.attention.AttentionBackendName` (`flash_2`, `flash_3`, `torch`). Selection happens via `model_utils.detect_attn_implementation()`.
- **HF checkpoint bridge** — `olmo_core.nn.hf.checkpoint.{load_hf_model, save_hf_model}`. This is how we ingest HF starting checkpoints (Qwen3, OLMo-2/3) and export trained models back to HF format ([scripts/train/convert_olmo_core_to_hf.py](../../scripts/train/convert_olmo_core_to_hf.py)).
- **LM head** — `olmo_core.nn.lm_head.{LMHead, LMOutputWithLoss}`, subclassed in `DPOLMHead` so per-token log-probs are computed inside the compiled module (avoiding DTensor/compile interactions in the outer loss code).
- **Parallelism** — `DataParallelType`, `TransformerDataParallelConfig`, `TransformerDataParallelWrappingStrategy.blocks`, `TransformerActivationCheckpointingConfig`, and TP / CP knobs on the train-module config. See [olmo_core_sharding.md](olmo_core_sharding.md).
- **Optim** — `AdamWConfig`, `CosWithWarmup`, `LinearWithWarmup`, the generic `OptimConfig` / `Scheduler` base classes, and recently `MuonConfig`.
- **Training infra** — `olmo_core.train.Trainer`, `olmo_core.train.train_module.TransformerTrainModule`, the `callbacks` package (`CheckpointerCallback`, `WandBCallback`, `CometCallback`, `ProfilerCallback`, `GPUMemoryMonitorCallback`).
- **Distributed utils** — `olmo_core.distributed.utils.is_distributed` and DTensor helpers. We initialise `torch.distributed` ourselves (or Ray does) and let OLMo-core build the device mesh.
- **Config primitives** — `olmo_core.config.DType` for typed dtype handling in configs.

### Model coverage

`OLMO_MODEL_CONFIG_MAP` in [olmo_core_utils.py](../../open_instruct/olmo_core_utils.py) lists every HF name that resolves to an OLMo-core `TransformerConfig`:

- OLMo-2: 1B (v2), 7B, 13B, 32B
- Olmo-3: 7B
- OLMoE: 1B-7B (mixture-of-experts)
- Qwen3: 0.6B, 1.7B, 4B, 8B, 14B, 32B (base and instruct)

Anything outside this map cannot be trained via the OLMo-core path until a matching `TransformerConfig` is added upstream in OLMo-core. This is the main *practical* constraint of choosing OLMo-core over HF Transformers.

### Initialisation pattern

Both DPO and GRPO use the "FSDP-first" load pattern (see [olmo_core_sharding.md](olmo_core_sharding.md#fsdp-first-loading-pattern)):

1. Build OLMo-core `Transformer` on CPU with freshly initialised weights.
2. Construct the `TrainModule`, which calls `parallelize_model()` and shards the (random) weights.
3. Call `load_hf_model()` to stream HF checkpoint weights directly into the *already-sharded* model.

This avoids ever holding a full unsharded copy of a 32B model on a single GPU, which is what makes scaling cheap.

### Checkpoint flow

- **In**: HF checkpoint → `load_hf_model()` → sharded OLMo-core model.
- **During training**: `CheckpointerCallback` writes OLMo-core-format distributed checkpoints (per-rank shards) on a schedule.
- **Out**: [scripts/train/convert_olmo_core_to_hf.py](../../scripts/train/convert_olmo_core_to_hf.py) reads a distributed checkpoint and calls `save_state_dict_as_hf` to emit HF-format weights + tokenizer + config — that's what gets uploaded for downstream eval / inference.

## How this compares to the alternatives

The "OLMo-core vs X" question only makes sense per-component. Open-instruct has three serious alternatives it could have used instead:

### vs. HuggingFace Transformers + DeepSpeed/Accelerate (what `grpo_fast.py` and `finetune.py` still use)

| Axis | OLMo-core path | HF + DeepSpeed path |
|---|---|---|
| Model definitions | Custom `Transformer`, tuned for FSDP2 + compile, small whitelist of supported models | `AutoModelForCausalLM`, ~every model on the Hub |
| Parallelism | FSDP2 / HSDP / TP via DTensor, node-aware auto-config | ZeRO-1/2/3 via DeepSpeed; FSDP1 via Accelerate (older path) |
| Mixed-precision | bf16 params + fp32 reductions, explicit | DeepSpeed handles internally, less control |
| `torch.compile` | First-class — model + loss are compiled, AC interacts with compile | Patchy, depends on HF model + DS interaction |
| Activation checkpointing | Budget mode (memory-budget knob) | Block-level on/off |
| MoE | Native OLMoE support with custom kernels | Slow / unsupported for OLMoE |
| Optimizers | Fused AdamW, Muon out of the box | Fused AdamW; Muon requires custom integration |
| Checkpoints | Distributed (per-rank) DCP, async writes | DS ZeRO checkpoints; HF Trainer flat .pt or safetensors |
| Logging | Native WandB / Comet / profiler callbacks | HF Trainer callbacks or hand-rolled |
| RL ergonomics | TrainModule subclass → loss is plain PyTorch; we already do this for DPO/GRPO | DeepSpeed engine API is awkward to call per-microbatch; weight sync to vLLM requires bespoke gather logic |
| Model coverage | OLMo-2/3, OLMoE, Qwen3 only | Anything on the Hub |
| Throughput on Ai2 hardware | Generally higher MFU on A/H100 nodes (especially at scale and for MoE) | Solid but lags on >8 GPU and on MoE |

The bottom line: HF + DeepSpeed is **broader** (any model), OLMo-core is **deeper** (better throughput, better MoE, better RL ergonomics) for the subset of models we actually train.

### vs. vanilla PyTorch FSDP2

You could in principle wire DPO/GRPO directly on `torch.distributed.fsdp` and skip OLMo-core entirely. What you'd lose:

- Pre-built `TransformerDataParallelConfig` with sensible wrapping policies — you'd be writing your own `ModuleWrapPolicy` per model.
- Activation-checkpointing-budget mode, which is non-trivial to implement.
- HF ↔ DTensor checkpoint bridges (`load_hf_model` / `save_hf_model`).
- The callback / trainer / profiler scaffolding — you'd be reinventing `Trainer`.
- OLMoE: the MoE-specific FSDP wrapping and routing kernels.

What you'd gain: zero pin to an external repo, model freedom. For a research codebase that only trains a known set of architectures, this trade is bad: you spend weeks reimplementing what OLMo-core already does.

### vs. torchtitan

[torchtitan](https://github.com/pytorch/torchtitan) is the closest analogue in spirit — a small, opinionated PyTorch training stack using DTensor + FSDP2 + TP + compile. The differences:

- torchtitan is **pre-training-focused**; its data loaders and loss code assume large-scale LM pretraining.
- torchtitan supports a small set of LLaMA-shaped models out of the box; OLMo-core supports the exact set we care about (OLMo, OLMoE, Qwen3).
- OLMo-core has the HF checkpoint bridge baked in. torchtitan has its own format.
- OLMo-core ships with Ai2-specific glue (Beaker, OLMoE kernels, Muon, float8). For an Ai2 RL codebase that's a feature, not a wart.

For someone *outside* Ai2 building a from-scratch RL stack, torchtitan + a thin HF-compat layer is a reasonable choice. Inside the OLMo ecosystem, OLMo-core wins on integration cost.

### vs. Megatron-LM / Megatron-Core

Megatron is the high-throughput baseline for very large models (TP + PP + SP + DP). It's strictly more capable on >32B + pipeline parallelism, but:

- The model code is its own dialect; HF-checkpoint interop is awkward.
- Pipeline parallelism, which is Megatron's biggest win, isn't useful at the scales open-instruct trains (≤32B, ≤16 nodes).
- Customising the loss for RL/DPO inside Megatron's pipeline runtime is painful.

OLMo-core covers 95% of what we'd use Megatron for, with much cleaner Python ergonomics. When (if) we train >70B with PP, the calculus changes.

## When to use which path in this repo

- **DPO, any model in the supported set** → `dpo.py`. No reason to fall back.
- **GRPO, multi-node, OLMo / Qwen3 model** → `grpo.py` (OLMo-core + Ray). This is the actively-developed path.
- **GRPO, smaller / single-node, or a model not in `OLMO_MODEL_CONFIG_MAP`** → `grpo_fast.py` (DeepSpeed). It's the legacy path but it works and has broader model coverage.
- **SFT** → use OLMo-core's own SFT scripts. `finetune.py` is deprecated and is kept around only for compatibility.
- **Adding a brand-new architecture** → if you can land a `TransformerConfig.<new_model>` in OLMo-core, do that and add it to `OLMO_MODEL_CONFIG_MAP`. Otherwise, train via the HF + DeepSpeed path.

## Surface area to remember when upgrading the pin

When bumping the OLMo-core commit in [pyproject.toml](../../pyproject.toml), things that historically break:

- `TransformerDataParallelConfig` field names / wrapping-strategy enum values.
- `TransformerTrainModule` constructor signature (especially around AC config and TP).
- Callback priorities (`BeakerCallbackV2` sets its priority relative to `CometCallback` / `WandBCallback`).
- `load_hf_model` / `save_hf_model` signature.
- Optimizer config classes (`AdamWConfig`, `MuonConfig`).
- LM head signature (`DPOLMHead` extends `LMHead.forward`).
- Parameter naming conventions — `olmo_core_to_hf_name` in [grpo_callbacks.py](../../open_instruct/grpo_callbacks.py) is the canonical place to update when OLMo-core's parameter names change for Qwen3 / LLaMA-shaped models.

If you touch any of the wrapper files listed above and the build breaks on the next image, that's the place to start.
