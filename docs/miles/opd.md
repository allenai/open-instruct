# Experimental Megatron OPD integration

This integration brings the Qwen OPD investigation onto the primary MILES
application layout. It uses **Megatron for training and SGLang for student and
teacher inference**, through `python -m open_instruct.miles`. GRPO continues to use
Core by default; [Megatron verifier GRPO](megatron-grpo.md) shares this runtime. Core OPD from the historical branch is not included in this integration.
A run file selects OPD with `training.algorithm = "opd"`; the default
`trainer.backend = "megatron"` routes it to `OPDRunSpec` instead of the GRPO
`RunSpec`.

## Provenance and qualification

- OPD investigation source: `c5f4f405607bbcfd441b5ca6f68edd1538389e1c`
  (`robertb/miles-qwen35-opd-async-audit`), first integrated on
  `robertb/miles-opd-integrated` at `3c5074c06aa9c523db6a8a3db0c02f0e1a29a5c8`.
- Kevin's source at the investigation fork: `77524d4f08efc3d1658edc39dee1648b976d4deb`.
- Runtime sources are the immutable commits in `runtime/miles/runtime.lock.json`,
  shared with GRPO; the old runtime patch bundles are not restored. The OPD route
  additionally installs a pinned `mbridge` revision for Megatron conversion.

The old branch and its run artifacts remain the historical record. The
investigation's selection, serving-agreement and repeated-evaluation scripts are
not part of this integration; they remain at `3c5074c06aa9` above. An
eight-update Qwen3.5 2B bridge run on this integration tracked that baseline's
reverse KL; it is a mechanics check, not a learning-parity qualification.

The application modules live in `open_instruct.miles.distillation`. Backend
dispatch is CPU-safe; importing the planner does not import Megatron or SGLang.
The async producer and measured queue extend the native MILES lifecycle instead
of copying the old Core adapter back into Open Instruct.

## Migrated correctness boundaries

- A publication wrapper returns the native weight version when it publishes;
  skipped publication returns `None`. The driver must propagate published
  versions to its rollout executor.
- Selection traces retain numeric span versions for historical analyses and
  also serialize the current runtime's per-call token spans.
- Queue instrumentation preserves the native `put` result, including an
  intentional filter drop. It records aborted work before retry resets it.
- Timing wraps the current driver's shared publication helper and executor.
- The trainer/rollout metric measures the **current forward**, even when PPO
  uses stored rollout scores as its anchor. This correction changes observation,
  not the loss or optimizer.
- Pre-rendered text remains text when Qwen supplies a multimodal processor.
- CP1 packed attention retains the separate padding-sequence layout needed
  by the Qwen/Blackwell FlashAttention path. The GPU test uses the current
  input-aligned loss-mask field.
- OPD evaluation calls the shared verifier with the historical whole-answer,
  nonzero-truncation protocol explicitly, independently of GRPO reward defaults.

## Policy loss

`distillation.policy_loss` picks the surrogate applied to the reverse-KL advantages.
The default `"ppo"` is MILES' clipped policy loss. `"dppo"` uses the
[Megatron verifier GRPO](megatron-grpo.md) DPPO trust-region mask
(`distillation.dppo_divergence_type`, default `"tv"`; `distillation.dppo_threshold`,
default 0.1), so an OPD arm and a verifier-RL arm can share one objective. DPPO
requires `distillation.use_rollout_logprobs = true` and rejects EOPD, TIS, OPSM,
a KL loss and an entropy bonus.

## Local and runtime checks

Use `open_instruct/test_miles_opd.py` and `open_instruct/test_miles_eopd.py` for
CPU configuration and objective checks. Run `tests/miles/test_opd*.py` in the
candidate image with `--require-miles-runtime`; the attention test requires a
GPU. `test_scoring_pass_loss.py` checks the current-forward metric and preserves
loss/gradient equivalence across scoring modes.

Copy `tests/miles/fixtures/opd/qwen35-4b-tiny.toml` into ignored `runs/` before changing it.
It is a mechanics fixture, not a qualified learning recipe. Keep
campaign-specific configurations and rendered specifications there. Launch via
the committed-image wrapper described in [launching](launching.md). CPU asset
preparation is unallocated on Saturn, with Jupiter as the scheduling fallback;
GPU experiment placement and limits come from the reviewed run specification.
