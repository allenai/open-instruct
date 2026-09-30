# Experimental Megatron OPD integration

This branch brings the Qwen OPD investigation onto the primary MILES application
layout. It uses **Megatron for training and SGLang for student and teacher
inference**, through `python -m open_instruct.miles`. GRPO continues to use Core.
Core OPD from the historical branch is not included in this integration.

## Provenance and qualification

- Application base: `c41226d841ec5da5abd81e4255187ca4c486cfce` (`robertb/miles-olmo-core`).
- OPD investigation source: `c5f4f405607bbcfd441b5ca6f68edd1538389e1c`
  (`robertb/miles-qwen35-opd-async-audit`).
- Kevin's source at the investigation fork: `77524d4f08efc3d1658edc39dee1648b976d4deb`.
- Runtime sources remain immutable commits in `runtime/miles/runtime.lock.json`;
  the old runtime patch bundles are not restored.

The old branch and its run artifacts remain the historical record. Neither its
GPU results nor primary-branch Core qualifications establish learning parity for
this combination. The integration must pass runtime checks and an eight-update
bridge run before the four controlled-exposure arms launch.

The application modules live in `open_instruct.miles.distillation`. Backend
dispatch is CPU-safe; importing the planner does not import Megatron or SGLang.
The async producer and measured queue extend the native MILES lifecycle instead
of copying the old Core adapter back into Open Instruct.

## Experiment invariants

The bridge repeats the completed eight-update async **drop** baseline: same
student and teacher checkpoints, prompt source and order seed, four learner GPUs
(TP2/DP2), three student GPUs, one teacher GPU, 128 prompts with two responses per
update, 16,384 response tokens, optimizer and token loss settings. It writes a
fresh output directory. The common evaluator uses the existing frozen panel and
identical per-response seeds. A runtime upgrade is the treatment; individual
changes within that upgrade are not isolated by this bridge.

The subsequent 2×2 compares the admitted versus selected frozen prompt cohorts,
and publication after every update versus every four updates. All four arms must
use one frozen image. The first arm gates the remaining three on exact cohort
membership/order, eight optimizer steps, expected policy ages, finite gradients,
correct teacher advantages, checkpoint completion and fresh export reload.
This is a control for exposure and policy age, not async throughput.

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
