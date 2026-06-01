# Qwen3-4B-Base RL-Zero · linear shaping α=1.0 · solve-rate warmup (latched, thr=0.3)

## Status

- **State**: training completed (1000 steps, exit 0) on the 2026-05-30 relaunch
- **Eval state**: submitted 2026-06-01 — oe-eval jobs for steps 100–1000 via `cse-579-scripts/submit_all_current_evals.sh` (priority normal, ai2/saturn+ceres). Fetch with `cse-579-experiments/fetch_eval_results.sh` once the jobs finish.
- **Last updated**: 2026-05-29

## Purpose

Tests a **competence-gated warmup** as an alternative to the step-based schedule
in [`qwen_4b_base_linear_alpha1_warmup_linear.md`](qwen_4b_base_linear_alpha1_warmup_linear.md).
Length pressure stays **off** (weight 0) until the model demonstrates it can
solve — once the batch mean solve rate first reaches `solve_rate_threshold=0.55`,
the penalty **latches on** at full strength for the rest of training.

Shaping method/decay is held identical to the collapse run (linear, α=1.0); the
only change is `warmup_type=solve_rate`. Hypothesis: gating on competence rather
than on a fixed step count adapts the schedule to how fast this particular run
learns to solve, so length pressure never arrives before reasoning exists.

### Threshold calibration (why 0.55, not 0.3)

The first launch used `solve_rate_threshold=0.3`, but checking the no-shaping
baseline's `batch/percent_solved_mean` in W&B (run `0u6g080p`) showed it is
**≥0.3 on 99.4% of steps and crosses 0.3 at step 1** — so a 0.3 gate latches
immediately and degenerates into the `constant` warmup that already collapsed.
Reason: Qwen3-4B-Base already solves much of the *mixed* batch from step 0 (final
`ifeval_correct_rate≈0.99`, `code≈0.62`, `math≈0.38`); the aggregate solve rate
starts ~0.37 and drifts to ~0.55 with no sharp inflection. The baseline's
first-crossing step by threshold: 0.45→step 45, 0.50→103, **0.55→197**, 0.60→228.
0.55 fires the latch ~step 200 (single batch), giving a real schedule difference
from constant and roughly matching run A reaching half-strength at step 250.
Caveat: even at 0.55 the aggregate gate mostly reflects the easy tasks, not the
hard reasoning (math/AIME) we most want to protect — a per-domain (math) gate
would be cleaner but needs a code change.

### Latch (this run relies on a code change)

`compute_warmup_weight`'s solve_rate path is now **monotonic**: `group_solve_rate`
is a noisy per-batch mean (over 64 prompts × 8 samples), and the original hard
0/1 switch would chatter on/off as that mean wobbled around 0.3. The latch
(`open_instruct/data_loader.py` + `length_reward_shaping.py`, branch
`ian/length-shaping-warmup` @ `fe367db0`) records that the threshold was reached
and keeps shaping on thereafter; latch state is persisted in the data-prep
actor's `get_state`/`set_state` so a preempted+resumed run doesn't un-latch.

## Beaker

### Attempts

| # | URL | Launched (UTC) | Terminated (UTC) | Exit | Notes |
|---|-----|----------------|------------------|------|-------|
| 1 | [01KSTQDR…](https://beaker.org/ex/01KSTQDR0J0JB179FH594XRN6D) | 2026-05-29 20:40 | 2026-05-29 20:51 | cancelled | thr=0.3 — latches at step 1 ≈ constant warmup; cancelled before training (see Threshold calibration) |
| 2 | [01KSTR2C…](https://beaker.org/ex/01KSTR2CJ3QF43NYGGKQJ1CPVY) | 2026-05-29 20:51 | 2026-05-30 ~21:50 | deleted | thr=0.55; reached ~step 379, experiment deleted by a colleague by accident. |
| 3 | [01KSXDZ2…](https://beaker.org/ex/01KSXDZ2TMB7K2PM813R1K2403) | 2026-05-30 21:51 | 2026-05-31 | 0 | Completed all 1000 steps (relaunch, fresh from step 0); thr=0.55. |

- **Workspace**: ai2/olmo-instruct
- **Cluster**: ai2/jupiter
- **Resources**: 1 node × 8 GPUs

## Configuration

- **Launch script**: `cse-579-scripts/length_shaping_rl_qwen.sh`
  (`SHAPING_METHOD=linear DECAY_PARAM=1.0 WARMUP_TYPE=solve_rate SOLVE_RATE_THRESHOLD=0.55`)
- **Branch / commit**: `ian/length-shaping-warmup` @ `fe367db0`
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl` (verifiable-rewards-only mix)
- **Shaping**:
  - method=linear, decay_param=1.0 (α)
  - warmup_type=solve_rate, solve_rate_threshold=0.55 (latched; see calibration above)
  - correctness_threshold=0.0 (auto-resolved; no format reward in this config)
  - use_raw_group_stats=false (matches the collapse run)
- **Other hyperparams**: lr=1e-6, total_episodes=512000 (~1000 steps),
  response_length=30720, pack_length=32768, num_samples=8, num_unique_prompts=64,
  deepspeed_stage=3, num_learners_per_node=4, vllm_num_engines=4
- **Image**: `nathanl/open_instruct_auto`, branch overlaid via in-container git-clone

## Outputs

- **exp_name**: `lenshape_qwen_4b_base_mixed_linear_p1.0_wsolve_rate`
- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wsolve_rate__<seed>__<ts>_checkpoints/`
  (steps 100 → 1000 every 100 steps; exact suffix assigned at runtime)
- **W&B run**: _(fill in once the run registers)_
- **Eval results path**: `cse-579-experiments/results/lenshape_qwen_4b_base_mixed_linear_p1.0_wsolve_rate/`

## Pair / baseline

- **Compare to**: [`qwen_4b_base_baseline.md`](qwen_4b_base_baseline.md) (no shaping),
  [`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md) (constant warmup,
  collapsed), and [`qwen_4b_base_linear_alpha1_warmup_linear.md`](qwen_4b_base_linear_alpha1_warmup_linear.md)
  (step-based warmup — the sibling of this run).

## Notes

The step at which the latch flips is itself a result: log `val/length_reward_warmup_weight`
(0 until the latch, 1 after) against `val/scores_pre_shaping` to see what solve
rate the model reached before length pressure turned on.

## Known issues

None yet. If `val/length_reward_warmup_weight` never leaves 0, the model never
reached a 0.55 batch solve rate — in that case the run is effectively an unshaped
baseline and the threshold needs lowering. Per the baseline trajectory the latch
should fire around step ~200; if it fires at step 1, the per-batch signal is
running hotter than the baseline and 0.55 is still too low.
