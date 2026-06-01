# Qwen3-4B-Base RL-Zero · linear shaping α=1.0 · step-based warmup (frac=0.5)

## Status

- **State**: training completed (1000 steps, exit 0) on the 2026-05-30 relaunch
- **Eval state**: submitted 2026-06-01 — oe-eval jobs for steps 100–1000 via `cse-579-scripts/submit_all_current_evals.sh` (priority normal, ai2/saturn+ceres). Fetch with `cse-579-experiments/fetch_eval_results.sh` once the jobs finish.
- **Last updated**: 2026-05-29

## Purpose

Tests whether a **step-based warmup** of the length penalty prevents the
reasoning collapse seen in [`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md)
(linear α=1.0, `constant` warmup, which reward-hacked to ~7-token outputs and
lost all of AIME pass@32 and 30pp of Minerva).

The shaping method/decay is held **identical** to that collapse run (linear,
α=1.0); the *only* change is `warmup_type=linear` with `warmup_fraction=0.5`, so
the shaping weight ramps 0→1 over the first **500 of ~1000 steps** and is at full
strength thereafter. Hypothesis: giving the model the first half of training to
learn that chain-of-thought raises the verifier reward — before length pressure
reaches full strength — lets it reach a higher solve rate first, so the eventual
length pressure shortens *correct* reasoning rather than collapsing it.

Clean attribution: any difference vs the constant-warmup run is due to the
schedule alone.

## Beaker

### Attempts

| # | URL | Launched (UTC) | Terminated (UTC) | Exit | Notes |
|---|-----|----------------|------------------|------|-------|
| 1 | [01KSTQDC…](https://beaker.org/ex/01KSTQDCJ9RF60W1Z5885BTEYE) | 2026-05-29 20:39 | 2026-05-30 ~21:50 | deleted | Reached ~step 139; experiment deleted by a colleague by accident. |
| 2 | [01KSXDZ0…](https://beaker.org/ex/01KSXDZ07VPFHHFBE3V3T2RMGX) | 2026-05-30 21:51 | 2026-05-31 | 0 | Completed all 1000 steps (relaunch, fresh from step 0). |

- **Workspace**: ai2/olmo-instruct
- **Cluster**: ai2/jupiter
- **Resources**: 1 node × 8 GPUs

## Configuration

- **Launch script**: `cse-579-scripts/length_shaping_rl_qwen.sh`
  (`SHAPING_METHOD=linear DECAY_PARAM=1.0 WARMUP_TYPE=linear WARMUP_FRACTION=0.5`)
- **Branch / commit**: `ian/length-shaping-warmup` @ `fe367db0`
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl` (verifiable-rewards-only mix)
- **Shaping**:
  - method=linear, decay_param=1.0 (α)
  - warmup_type=linear, warmup_fraction=0.5 (full strength by step ~500)
  - correctness_threshold=0.0 (auto-resolved; no format reward in this config)
  - use_raw_group_stats=false (matches the collapse run, so advantage
    normalization isn't a confound)
- **Other hyperparams**: lr=1e-6, total_episodes=512000 (~1000 steps),
  response_length=30720, pack_length=32768, num_samples=8, num_unique_prompts=64,
  deepspeed_stage=3, num_learners_per_node=4, vllm_num_engines=4
- **Checkpoint state dir**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wlinear_1780087169`
- **Image**: `nathanl/open_instruct_auto`, branch overlaid via in-container git-clone

## Outputs

- **exp_name**: `lenshape_qwen_4b_base_mixed_linear_p1.0_wlinear`
- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wlinear__<seed>__<ts>_checkpoints/`
  (steps 100 → 1000 every 100 steps; exact suffix assigned at runtime)
- **W&B run**: _(fill in once the run registers)_
- **Eval results path**: `cse-579-experiments/results/lenshape_qwen_4b_base_mixed_linear_p1.0_wlinear/`

## Pair / baseline

- **Compare to**: [`qwen_4b_base_baseline.md`](qwen_4b_base_baseline.md) (no shaping)
  and [`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md) (same shaping,
  constant warmup — the collapse this run is trying to avoid).

## Notes

_(running)_

## Known issues

None yet. Watch `val/scores_pre_shaping` vs `val/scores_post_shaping`: with the
ramp, the post-vs-pre gap should open gradually over steps 0–500 rather than
immediately, and `val/sequence_lengths_solved` should drop later and less
steeply than the constant-warmup run.
