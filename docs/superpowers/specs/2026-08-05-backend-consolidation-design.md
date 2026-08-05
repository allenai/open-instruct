# Backend Consolidation: OLMo-core as Primary, DeepSpeed Deprecated

**Date:** 2026-08-05
**Status:** Approved design, pre-implementation
**Branch:** `backend-parity`

> **HARD GATE:** No Part 2 work (renames, moves, shims, doc rewrites) begins until
> Part 1 produces its per-stage verdicts. A stage that has not passed its A/B does
> not get touched. Part 1 is the only work authorized right now.

## Problem

open-instruct carries two training backends for each of SFT, DPO, and GRPO:

| Stage | DeepSpeed/Accelerate | OLMo-core |
|---|---|---|
| SFT | `open_instruct/finetune.py` | `open_instruct/olmo_core_finetune.py` |
| DPO | `open_instruct/dpo_tune_cache.py` | `open_instruct/dpo.py` |
| GRPO | `open_instruct/grpo_fast.py` | `open_instruct/grpo.py` |

The docs call OLMo-core "recommended" for all three stages, but:

- No same-model, same-config, cross-backend throughput or loss comparison has ever
  been published for any stage. Existing evidence is confounded (different models,
  batch sizes, images).
- Naming is misleading: `grpo_fast.py` is the *older* DeepSpeed path; `grpo.py` is
  the newer OLMo-core one. `dpo_tune_cache.py` vs `dpo.py` similarly hides which is
  which.
- CI tests a different backend per stage (SFT: DeepSpeed, DPO: OLMo-core,
  GRPO: DeepSpeed) and never both.
- Both OLMo-core backends *are* in heavy production use (44 Qwen3-30B SFT runs,
  177 Qwen3-4B GRPO runs in Beaker workspaces, July 2026) but this is invisible in
  the committed scripts.

Goal: gain measured confidence in the OLMo-core backend per stage, then reorganize
so OLMo-core owns the primary names and DeepSpeed is visibly deprecated.

## Decisions (made with user)

1. **Ordering:** Confidence first, naming later. Benchmarks gate the renames.
2. **Confidence bar:** step-level parity + throughput. Matched-config A/B runs
   (~150 steps), OLMo-core must win or tie on MFU and tokens/sec/GPU, and
   `train_loss` must track within ~1%. Full-training + downstream evals are NOT
   required for the rename.
3. **Naming scheme:** plain names for OLMo-core, `_deepspeed` suffix for legacy,
   with the shared GRPO harness extracted first so `grpo_deepspeed.py` is honestly
   DeepSpeed-only.
4. **Compatibility:** shim modules at the old paths for one release (re-export +
   `DeprecationWarning`). Work happens on branch `backend-parity` → PRs; shim
   removal decided later.
5. **Models:** per-stage models (not one model everywhere). The variety is
   deliberate coverage: the two backends run different model *implementations*
   (HF `AutoModelForCausalLM` vs OLMo-core `TransformerConfig` + HF weight load),
   so each pair validates a distinct architecture + weight-conversion path.

## Part 1 — Confidence (benchmarks)

### Pre-work (code changes required before runs can be measured)

1. **`PerfCallback` for OLMo-core SFT.** `olmo_core_finetune.py` builds callbacks
   via `build_base_callbacks` only and emits no `perf/mfu_step` or
   `perf/tokens_per_second_*`. Mirror the wiring in `dpo.py:80`. (~8 lines)
2. **Per-log-period TPS for `finetune.py`.** Its `per_device_tps` is cumulative
   since process start, burying steady state. It already tracks
   `total_tokens_this_log_period`; add the time delta. (~4 lines)
3. **Flag guard in `grpo.py`.** Reject the six DeepSpeed-only flags it silently
   ignores (`deepspeed_stage`, `deepspeed_zpg`, `deepspeed_offload_param`,
   `deepspeed_offload_optimizer`, `deepspeed_checkpoint_load_universal`,
   `sequence_parallel_size`), mirroring the guard pattern at `dpo.py:142`.

### The six runs

New scripts under `scripts/train/debug/backend_ab/` (do not edit historical sweep
scripts). All six built from ONE commit into ONE image via
`build_image_and_launch.sh`.

| # | Stage | Backend | Entrypoint | Model | Hardware |
|---|---|---|---|---|---|
| 1 | SFT | DeepSpeed | `finetune.py` | OLMo-2-1124-7B | 2 nodes × 8 H100 |
| 2 | SFT | OLMo-core | `olmo_core_finetune.py` | OLMo-2-1124-7B | 2 nodes × 8 H100 |
| 3 | DPO | DeepSpeed | `dpo_tune_cache.py` | Olmo-3-Hybrid-7B | 4 nodes × 8 H100 |
| 4 | DPO | OLMo-core | `dpo.py` | Olmo-3-Hybrid-7B | 4 nodes × 8 H100 |
| 5 | GRPO | DeepSpeed | `grpo_fast.py` | Qwen3-4B-Base | 1 node × 8 H100 |
| 6 | GRPO | OLMo-core | `grpo.py` | Qwen3-4B-Base | 1 node × 8 H100 |

Base configs: #1/#2 from `scripts/train/olmo2/finetune_7b.sh` +
`scripts/train/debug/oc_sft_multinode.sh` (flag renames:
`dataset_mixer_list`→`mixer_list`, `num_train_epochs`→`num_epochs`); #3/#4 from
the two `olmo-hybrid/7b_instruct_dpo_sweep*.sh` scripts; #5/#6 from the two
`qwen/qwen3_4b_dapo_math*.sh` scripts.

### Controls (each fixes a known confound)

- **One image, one commit** for all six (the two DPO sweeps default to different
  images today).
- **Identical checkpoint objects** per pair. Verify the weka path
  `HYBRID_INSTRUCT_SFT_0218_2.5e-5/step3256-hf` and HF
  `allenai/Olmo-Hybrid-Instruct-SFT-7B` are the same weights before launching the
  DPO pair.
- **Single LR per pair** (DPO pinned to 1e-6; drop the 5-LR sweep).
- **`--max_train_steps 150`**, discard first ~20 steps (compile + warmup).
- **Step-0 loss assertion per pair**: same weights + same first batch ⇒ initial
  loss must match closely (cf. PR #1620: byte-identical step 0). Divergence means
  the HF→OLMo-core conversion is wrong and throughput comparison is moot.
- **DPO**: report steady-state s/step, never job wall clock (`dpo.py` warm
  reference-logprobs cache on shared weka makes reruns look fast).
- **GRPO**: compare `time/training` and `learner_tokens_per_second_*` only;
  `time/total` is generation-bound and preemption-noisy (observed 11h45m–4d10h on
  nominally similar runs).

### Metrics compared

- `perf/mfu_step` (or equivalent), `perf/tokens_per_second_per_gpu`
- `train_loss` curve, delta < ~1% after warmup
- step-0 loss (conversion check)

### Verdict + publication

Per stage: OLMo-core wins/ties throughput AND loss tracks ⇒ stage passes.
Results table committed to `docs/algorithms/backend_comparison.md` with Beaker and
wandb links for all six runs (so results don't evaporate into Slack like the
Feb 2026 hybrid DPO numbers).

**Failure branch:** a stage that fails does NOT get renamed; instead file the gap
found and correct the docs to say DeepSpeed is primary for that stage. Either
outcome ends the misleading state.

Estimated cost: ~160–260 GPU-hours total, verdict in ~2 days.

## Part 2 — Reorganization (gated per-stage on Part 1)

### Target layout

```
open_instruct/
  sft.py                  # was olmo_core_finetune.py
  dpo.py                  # unchanged
  grpo.py                 # unchanged
  grpo_harness.py         # extracted from grpo_fast.py (shared RL harness)
  sft_deepspeed.py        # was finetune.py
  dpo_deepspeed.py        # was dpo_tune_cache.py
  grpo_deepspeed.py       # was grpo_fast.py, DeepSpeed trainer + main only
  finetune.py             # shim → sft_deepspeed (1 release)
  olmo_core_finetune.py   # shim → sft (1 release)
  dpo_tune_cache.py       # shim → dpo_deepspeed (1 release)
  grpo_fast.py            # shim → grpo_deepspeed (1 release)
```

### The grpo_fast split (prerequisite for the GRPO rename)

`grpo.py` imports 8 symbols from `grpo_fast.py` (`make_tokenizer`,
`setup_datasets`, `initialize_tools_and_envs`, `create_generation_configs`,
`build_base_env_config`, `setup_runtime_variables`, `EXCLUDED_ENV_VARS`,
`ModelGroup`). Extract these + related shared code (datasets, tokenizer,
vLLM/tool/reward setup, queues) into `grpo_harness.py` as a pure-move refactor —
both entrypoints must pass existing integration tests before the rename commit.
Only `PolicyTrainerRayProcess` (~660 lines) + the DeepSpeed `main` stay in
`grpo_deepspeed.py`.

### Order of stage PRs

1. **DPO** — smallest; `dpo.py` imports nothing from `dpo_tune_cache.py`.
   ~9 launch scripts + docs + CI regex to update.
2. **SFT** — same shape plus `olmo_core_finetune.py` → `sft.py`. The
   `finetune.py` shim matters most (tulu docs, external users).
3. **GRPO** — harness extraction commit, then rename commit (two reviewable
   commits in one PR).

### Per-stage PR contents

- `git mv` + shim modules (re-export `*` + `main`, module-level
  `DeprecationWarning` naming the replacement).
- Fix the existing `finetune.py` deprecation warning target (currently points to
  the OLMo-core repo instead of the in-repo replacement).
- Update all in-repo launch scripts, `docs/algorithms/*.md` (legacy path demoted
  to a "Legacy (DeepSpeed)" section; benchmark table linked as evidence),
  `.github/workflows/beaker-experiment.yml` (changed-file regexes + experiment
  matrix so CI tests the primary path), `CLAUDE.md` test-script listing,
  `pyproject.toml` lint-exclusion list, CHANGELOG entry with old→new table.
- Historical reproduction docs (`docs/tulu1_tulu2.md`, `docs/tulu3.md`,
  `docs/olmo2.md`) keep old commands verbatim — they document past runs; shims
  keep them functional for one release.

## Out of scope (deletion blockers, tracked separately)

- LoRA/QLoRA (DeepSpeed-only; no OLMo-core equivalent)
- Llama / Qwen2.5 model coverage (absent from `OLMO_MODEL_CONFIG_MAP`)
- `reward_modeling.py` (no OLMo-core version exists)
- DeepSpeed Ulysses sequence parallelism (`grpo_fast.py` only)
- Deleting any DeepSpeed module or removing shims

## Testing

- Part 1 pre-work: unit-testable where cheap (flag guard); metric changes verified
  on the benchmark runs themselves.
- Part 2: shims import-tested; both GRPO entrypoints pass existing integration
  tests after the harness extraction; per-stage CI experiment must pass on the
  renamed path before merge.
