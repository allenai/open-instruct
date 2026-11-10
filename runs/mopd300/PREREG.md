# MOPD 300: pre-registration (written 2026-11-10 PT, before any GPU job for this extension)

Kevin, 2026-11-10: "let's do mopd 300 too". This approval covers this bounded extension and its matched evaluations only.

## Why not the experiment ledger

The ledger (`allenai/olmo-post-training-ledger`) records experiments on registered Olmo bases. Qwen3.5-9B is not a registered base there, and the MOPD study was never ledgered. This file follows the ledger's rules instead:
- written before the launch, with the commit timestamp as proof;
- a question, predictions, falsifiers and decision rules fixed in advance;
- mechanical gates;
- no exclusions.

## Question

The original MOPD student was trained for 100 updates. Continue each of its three seeds to **300 student updates in total** (100 already done plus 200 more), with nothing else changed. Then ask:

At an equal number of **student** updates (300 vs 300), and on the same prompts in the same per-seed order, is MOPD ahead of, behind, or unclear relative to the existing mixed RL 300 on the internal math, IF and code evals?

Secondary: does MOPD improve from 100 to 300 student updates?

## What this is and is not

- **Not compute-matched.**
  - MOPD 300 = 300 student updates plus 3 × 100 frozen teacher updates (sunk) = **600 total updates**.
  - Mixed RL 300 = 300 updates.
  - This is a student-update-matched comparison only; with teachers counted, MOPD uses twice the updates. Every report must state both counts.
- **Single change:** the student's horizon (`training.num_rollouts` 100 → 300). Nothing else changes:
  - Same frozen teachers: math, IF and code, each the 100-update `hf-99`.
  - Same data (v4 `mix_all3`) and seeds (42, 43, 44).
  - Same recipe: DPPO TV 0.03, constant LR 1e-6, no warmup, 64 × 8, domain-balanced loss.
  - Same evaluation settings.
  - Same image (`01M5XNG1GRGN7GJ9X15TW9AFH4`), with a code overlay whose wrapper code equals cb457390b.
- **Continuation, not a weight-only restart.** Each run resumes from the original run's update-99 Megatron checkpoint: weights, optimizer and RNG state, plus the data cursor `global_dataset_state_dict_99.pt`, through the wrapper's native resume. `--use-checkpoint-opt-param-scheduler` is set on resume.
- **Why the LR schedule is unchanged:** the LR is constant with 0 warmup, and `lr-decay-iters` is not set for a constant schedule. The LR is therefore 1e-6 at every step, regardless of the horizon.
- **Root layout:** new output roots `mopd-qwen35-9b-all3-300u{,-s2,-s3}` are pre-seeded as in the Phase 1 extensions (decision log 2026-10-18). One deviation: `checkpoints/iter_0000099` is a real directory (hardlinks or a copy), not a symlink, because `keep_checkpoints = 1` retention calls `shutil.rmtree` on older iterations and would fail on a symlink. The original roots are not written to.
- **Known boundary effect, as in Phase 1:**
  - At update 100, fully-async in-flight groups are rebuilt from the saved ledger.
  - W&B starts a new run.
  - There is no eval at the resume point; the first new in-training eval is at update 109.

## Primary metrics and analysis (existing contract, unchanged)

- **Evals:**
  - Internal evals of `hf-299`: 256 held-out prompts per task, 8 answers per prompt, temperature 1, v4 graders, every answer saved.
  - The config is a copy of `all3-eval8-qwen35-9b-mixrl-hf299-s3.toml`, changing only the model source, name and roots.
- **Statistics:**
  - Script: `oi-mopd-phase1-runs/paired/seedstats3.py`, unchanged apart from a new setup `mopd300` and two pairs.
  - Method: two-level bootstrap, 95%, prompts resampled with shared indices, seeds resampled within setup, 4,000 resamples.
- **Comparisons:**
  1. **Primary:** MOPD 300 − mixed RL 300 (3 vs 3 seeds), for each of math, IF and code.
  2. **Secondary:** MOPD 300 − MOPD 100 (3 vs 3).
- **Decision rule per task:**
  - **ahead** if the 95% lower bound > 0;
  - **behind** if the upper bound < 0;
  - **unclear** otherwise. Unclear is not evidence of equality; no equivalence test is run.
- **No new margin.** The Oct 22 plan's rule ("MOPD +0.015 on IF without losing more than 0.02 on math") is reported descriptively only, as in the original study. It is not used as a decision rule.

## Predictions and falsifiers (fixed before any MOPD 300 result exists)

The basis is in-training 1-answer curves (≈ ±0.03): MOPD levelled off by about update 50, while mixed RL kept rising to 300.

- **P1:** MOPD 300 will **not** be ahead of mixed RL 300 on IF.
  - **F1:** if the IF lower bound of MOPD 300 − mixed RL 300 is > 0, the reading "MOPD levels off and mixed RL catches up" is falsified at 300 student updates.
- **P2:** MOPD 300 − MOPD 100 on IF will be small. **F2:** if its lower bound is > 0, the reading "MOPD has plateaued by about 100 student updates" is falsified.
- **Math and code:** no directional prediction; reported under the same rule.

## Matched secondary evaluations (descriptive, no significance tests, as for mixed RL 300)

- Standard benchmarks for MOPD 300 seeds 1–3, with the same recipe as the Nov 4 sweep:
  - olmo-eval-internal fa892c1;
  - AIME24/25 `pass_at_32_rlzero`, IFEval ×8, LiveCodeBench ×8;
  - temperature 1, 16,384 tokens, thinking off.
- The boxed-prompt AIME diagnostic, with the same wrapper and hash as Nov 6.
- Original-prompt AIME stays the primary AIME number.

## Gates (mechanical; pass or fail on artifacts and counters only)

- **G0 (before launch):**
  - Original roots: `workflow.json` complete, `latest_checkpointed_iteration.txt` = 99, and `iter_0000099` present with optimizer and RNG state.
  - Data cursor 99 present; teachers `hf-99` present.
  - Target roots absent.
  - New configs differ from the originals only in `name`, `training.num_rollouts` and `output.root`, plus comments.
  - `validate` passes.
- **G1 (startup):**
  - The log shows the checkpoint loaded at iteration 99 from the new root, and the data cursor loaded from `global_dataset_state_dict_99.pt`.
  - The first trained rollout id is 100.
  - The first-step train KL and reverse KL are in the original run's late range, i.e. there was no reset.
  - LR is 1e-6.
- **G2 (end):**
  - 300/300 rollouts, finite.
  - `hf-299` exported, and the end-of-run audit passes. An audit failure gets diagnosed as a possible audit-path bug; it is never a reason to discard the run.
  - Original roots unmodified: the mtimes in `iter_0000099` match the seed record.
- **G3 (evals):** 256 × 8 per task, 0 failed samples.

## Rules

- **No exclusions:** all three seeds are reported.
- **Mechanical failures:** a crash or preemption may be resumed or relaunched from the newest save with an identical config. A score-driven retry or extra arm is never allowed.
- **No other changes:** no new teachers, mixture tuning, new arms or broader sweeps.
- **Existing claims stay as written:** MOPD 100 vs mixed RL 100 and 300, the splits and IF weighting. MOPD 300 adds a row; it does not replace any.

## Spend estimate (incremental)

| Item | GPU-h |
|---|---|
| Training: 3 × 200 updates at ≈3.7 min/update on 8 GPUs (MOPD s1–2 measured ≈6.2 h per 100 updates) | ≈300 |
| Startup overhead and possible preemption replay | ≈+30 |
| Internal evals: 3 × 8 GPUs × ≈1–1.5 h | ≈30 |
| Standard benchmarks: 3 jobs × 2 GPUs | ≈15 |
| Boxed AIME: 3 jobs × ≈0.9 GPU-h | ≈3 |
| **Total** | **≈380** |

- Peak 24 GPUs (3 × 8) on Holmes; ≤ 64 GPUs.
- Wall clock ≈12–14 h of training after the jobs start, plus queueing.
