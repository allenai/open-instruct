# Models and checkpoints

The MILES path uses an HF descriptor/tokenizer for SGLang and constructs the
training model through the Core adapter. Live weight publication exports tensors
in serving layout; it does not write and reload an HF directory every update.
There is no Megatron checkpoint conversion in this path.

| Model family | Current evidence and route |
|---|---|
| Conventional Olmo MoE | Tiny local and EP2 lifecycle/parity checks; choose a compatible HF descriptor |
| Earlier KDA/latent MoE SFT model | Primary full-SFT GSM8K comparison and async/replay evidence on B300 |
| Small hero, new attention features | Adapter/conversion work exists; see [hero support](hero-support.md) for exact scope before selecting it |
| Dense Olmo 3 | Separate FSDP adapter; two-GPU 7B smoke and audit passed, with zero-advantage batches. Nonzero-gradient full-model learning/resume remain unqualified; see [qualification](olmo3-pre-rl.md) |
| Other HF architectures | Do not infer support from safetensors or tokenizer compatibility; inspect the model factory or use an existing GRPO implementation |

For the authoritative factories inspect `open_instruct/miles/models.py`,
`moe_models.py`, and `standard_models.py`. Both native `grpo.py` and the MILES path
use OLMo-core, but have different orchestration and serving implementations.

## Inputs

`model.format="hf"` requires config.json, safetensors weights and a usable
compatible tokenizer/chat template. `model.format="olmo_core"` also requires
`model.hf_template` for export. `conversion.hf_output` is the prepared HF directory;
Core conversion currently disables forward validation, so conversion success is
not a numerical or routing-parity certificate.

Preserve router storage/computation precision and architecture metadata. The
KDA/latent runs depend on BF16 stored routers with FP32 router computation; inspect
[precision evidence](measurements/router-precision-20260911.md) before adopting a
new export. Do not blindly cast a checkpoint to address a mismatch.

Preparation treats sources as read-only. HF staging references source weights:
retain the source for the run's lifetime. Paths, descriptor, task rendering and
chat template determine the effective model input; matching a weight filename
alone is insufficient for a comparison.

## Save, resume and export

Core writes native distributed checkpoints synchronously. `training.save_interval`
counts collections; use one optimizer update per collection if comparing that
cadence directly with update numbers. `output.export_hf=true` requests a final HF
export, separate from the recoverable native optimizer checkpoint. Megatron
`async_save` and retention settings are not substitutes.

`launch.auto_resume=true` permits supported Beaker retries. Workflow preparation
is reused only for the same recorded specification; the latest completed native
checkpoint is loaded when present, otherwise training restarts from the input.
Multi-node/managed configurations currently require auto_resume=false. Manual
relaunch with the same unchanged specification/output root can adopt a completed
checkpoint after failure; a completed run root cannot be overwritten. Do not
change the recipe in-place and call it a resume.

Budget checkpoint disk space, synchronous save time and startup load time. See
[checkpoint measurements](measurements/checkpoint-perf-20260911.md) for the measured
EP2 workload, and [operations](operations.md) for artifact/completion checks.
