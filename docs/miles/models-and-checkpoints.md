# Models and checkpoints

The MILES path uses an HF descriptor/tokenizer for SGLang and constructs the
training model through the Core adapter. Live weight publication exports tensors
in serving layout; it does not write and reload an HF directory every update.
There is no Megatron checkpoint conversion in this path.

| Model family | Current evidence and route |
|---|---|
| Conventional Olmo MoE | Tiny local and EP2 lifecycle/parity checks; choose a compatible HF descriptor |
| Earlier KDA/latent MoE SFT model | Primary full-SFT GSM8K comparison and async/replay evidence on B300 |
| Small hero, new attention features | Paired SFT four-update barrier mechanics passed on H100 EP4 + TP1 with automatic fused rounding; see [hero support](hero-support.md) for evidence and remaining limits |
| Dense Olmo 3 | Separate FSDP adapter; two-GPU 7B smoke, fresh-process resume, HF export and fresh serving reload passed, with zero-advantage batches. Nonzero-gradient full-model learning remains unqualified; see [qualification](olmo3-pre-rl.md) |
| Other HF architectures | Do not infer support from safetensors or tokenizer compatibility; inspect the model factory and identify any support gap before choosing a backend; deprecated GRPO is not an automatic fallback |

For the authoritative factories inspect `open_instruct/miles/models.py`,
`moe_models.py`, and `standard_models.py`. Both deprecated `grpo.py` and the MILES path
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

Native checkpoints roll by default: after each commit, only the newest
`core.checkpoint_keep_last` committed checkpoints (default 1) and every
`core.checkpoint_keep_every`-th completed update (unset by default) remain; the
others and their MILES cursors are deleted, interrupted saves are left alone, and
the final save is always the newest. Set `core.checkpoint_keep_last` to a larger
count, or explicitly unset it, to retain more. Retention never touches a
checkpoint loaded from another run's root.

`launch.auto_resume=true` permits supported Beaker retries. Workflow preparation
is reused only for the same recorded specification; the latest completed native
checkpoint is loaded when present, otherwise training restarts from the input.
Multi-node configurations may keep auto_resume=true; a restart into the same output root adopts the newest completed checkpoint and continues the rollout cursor, qualified in [multi-node resume](measurements/multinode-resume-20260922.md). Forced preemption, and restarts of runs carrying a managed judge, remain unqualified. Manual
relaunch with the same unchanged specification/output root can adopt a completed
checkpoint after failure; a completed run root cannot be overwritten. Do not
change the recipe in-place and call it a resume.

Budget checkpoint disk space, synchronous save time and startup load time. See
[checkpoint measurements](measurements/checkpoint-perf-20260911.md) for the measured
EP2 workload, and [operations](operations.md) for artifact/completion checks.

## Using a template that does not ship with the checkpoint

Point `model.hf_template` at a tokenizer directory, for example a local snapshot of
`allenai/dolma2-tokenizer-olmo35`, and preparation stages that directory's tokenizer,
chat template and stop-token `generation_config.json` over the checkpoint's own. The
vocabulary must match the checkpoint's; the Olmo 3.5 tokenizer is vocabulary-identical
to the base dolma2 tokenizer, so only the template changes. Both the trainer's scoring
and the serving engines read the staged directory, so prompts, stop tokens and
log-probabilities all follow the override. Runs prepared under different templates are
different conditions: compare them only through a fresh evaluation, which
`scripts/miles/gsm8k_test_eval.py --tokenizer DIR` renders and serves with the same
override.

```toml
[model]
source = "/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf"
format = "hf"
hf_template = "/weka/oe-training-default/robertb/open-instruct/tokenizers/dolma2-tokenizer-olmo35"
```
