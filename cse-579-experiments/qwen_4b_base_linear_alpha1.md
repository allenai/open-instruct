# Qwen3-4B-Base RL-Zero · linear shaping, α=1.0

## Status

- **State**: training completed (attempt 2; HF push failed but training and checkpoints were fine)
- **Eval state**: retrieved (5 oe-eval beaker jobs for step_1000 succeeded; results saved under `cse-579-experiments/results/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000/`)
- **Last updated**: 2026-05-11

## Purpose

First real test of dynamic length-aware reward shaping. Pairs against Jacob's
`baseline_rl.sh` (no shaping, otherwise identical config) to answer:

1. Does linear α=1.0 train stably from Qwen base without diverging?
2. Does mean response length on solved problems decrease vs the unshaped baseline?
3. Does AIME 2025 / Minerva Math pass-rate hold up at the shorter lengths?

α=1.0 is the proposal's middle-of-the-road decay strength: a correct response
2× the length of the shortest correct response in its group is zeroed out.

## Beaker

### Attempts

| # | URL | Launched (UTC) | Terminated (UTC) | Exit | Notes |
|---|-----|----------------|------------------|------|-------|
| 1 | [01KQTD4D…](https://beaker.org/ex/01KQTD4DJ57C1SY8A1MFNS3GFC) | 2026-05-04 21:08 | 2026-05-04 22:36 | 1 | `checkpoint_state_dir` validation failure at parse time (see Known issues) |
| 2 | [01KQTJDA…](https://beaker.org/ex/01KQTJDAE5J37VZ0VRXKEHGWTY) | 2026-05-04 22:40 | 2026-05-05 02:24 | 1 | Training reached step 1000 (all 512000 episodes); checkpoints saved. Exit 1 was from final HF push to `allenai/open_instruct_dev` (403 — token lacks write to allenai org). Auto-launched evals at steps 800/900/1000 all failed due to a tokenizer-compat bug (see Known issues). |

- **Workspace**: ai2/olmo-instruct
- **Cluster**: ai2/jupiter
- **Resources**: 1 node × 8 GPUs

## Configuration

- **Launch script**: `cse-579-scripts/length_shaping_rl_qwen.sh`
- **Branch / commit**: `ian/length-shaping` @ `50cd7ccd`
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl` (verifiable-rewards-only mix)
- **Shaping**:
  - method=linear, decay_param=1.0 (α)
  - warmup_type=constant (full strength from step 0)
  - correctness_threshold=0.0 (auto-resolved; no format reward in this config)
- **Other hyperparams**: lr=1e-6, total_episodes=512000, response_length=30720,
  pack_length=32768, num_samples=8, num_unique_prompts=64, deepspeed_stage=3,
  num_learners_per_node=4, vllm_num_engines=4
- **Image**: `nathanl/open_instruct_auto`, with the `ian/length-shaping` branch
  overlaid via the in-container git-clone step

## Outputs

- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant__1__1777934949_checkpoints/`
  (steps 100 → 1000 every 100 steps)
- **W&B run**: https://wandb.ai/ai2-llm/open_instruct_internal/runs/2pkl9fhp
- **Eval beaker jobs** (submitted manually via `cse-579-scripts/submit_lenshape_qwen_eval_jobs.sh` on 2026-05-11):
  - alpaca_eval_v3: [01KRCA5XQS5JV19DV7GHM0T7PE](https://beaker.org/ex/01KRCA5XQS5JV19DV7GHM0T7PE)
  - minerva_math_500: [01KRCA5YJA5W0VSNZ90BQKGF75](https://beaker.org/ex/01KRCA5YJA5W0VSNZ90BQKGF75)
  - ifbench::tulu: [01KRCA5ZC2NBWKSGAYT94QQ3F5](https://beaker.org/ex/01KRCA5ZC2NBWKSGAYT94QQ3F5)
  - livecodebench: [01KRCA605NWVCD39BK0PP7JSTT](https://beaker.org/ex/01KRCA605NWVCD39BK0PP7JSTT)
  - aime 2025 pass@32: [01KRCA615TJX7DNCQAGJPGHR2Q](https://beaker.org/ex/01KRCA615TJX7DNCQAGJPGHR2Q)
- **Eval results path**: `cse-579-experiments/results/lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000/` — saved by `cse-579-experiments/fetch_eval_results.sh` (one `metrics.json` + `length_stats.json` per task). Regenerate the summary table below with `uv run python cse-579-experiments/summarize.py lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000`.

## Pair / baseline

- **Compare to**: [`qwen_4b_base_baseline.md`](qwen_4b_base_baseline.md) — same
  RLVR recipe, no length shaping. Results retrieved 2026-05-11.

### Headline contrast (baseline → treatment, step_1000)

| Task | Baseline | Treatment | Δ acc | Length compression (tok mean) |
|------|----------|-----------|-------|--------------------------------|
| Minerva exact_match_flex | 54.2% | 23.6% | **−30.6pp** | 1158 → 7.8 (148×) |
| AIME pass_at_32 | 16.7% | 0.0% | −16.7pp | 1840 → 7.4 (250×) |
| LiveCodeBench pass_at_1 | 7.7% | 3.6% | −4.1pp | 174 → 25 (7×) |
| ifbench wildchat | 73.3% | 69.8% | −3.5pp | 424 → 54 (8×) |
| ifbench wildchat OOD | 55.7% | 55.6% | ~0 | 768 → 176 (4×) |
| ifbench OOD | 43.7% | 40.3% | −3.4pp | 718 → 124 (6×) |
| AlpacaEval LC winrate | 6.56 | 5.98 | −0.6 | 1067 → 152 (7×) |

The interpretation under "Results" below is now supported by the baseline:
the baseline run **does** produce real chain-of-thought reasoning (mean 1158
tokens on Minerva, 1840 on AIME, with ✓ shorter than ✗ as you'd expect) and
gets meaningful accuracy on hard reasoning. Our linear α=1.0 treatment
collapses that into 7-token boxed answers and loses 30pp on Minerva and all
of AIME pass@32.

## Results (step_1000)

Numbers below are pulled programmatically from the saved metrics in
`results/<run_dir>/<task>/`. Regenerate with:

```
uv run python cse-579-experiments/summarize.py lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000
```

(The table is regenerated from disk; don't edit the numbers by hand. Token
counts come from `model_output[*].num_tokens` — the same unit the reward
shaping operated on during training.)

<!-- BEGIN: summarize.py output for this run; do not edit manually -->

### `lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000`

| Task | Primary | Items (✓/✗/?) | Subset | gens | Tok mean | Tok std | Tok p50 | Tok p90 |
|------|---------|----------------|--------|------|----------|---------|---------|---------|
| `aime` | pass_at_1=0 | n=30 (✓ 0, ✗ 30, ? 0) | **all** | 960 | 7.4 | 0.8 | 7 | 8 |
| ↳ correct | | | **✓** | 0 | – | – | – | – |
| ↳ incorrect | | | **✗** | 960 | 7.4 | 0.8 | 7 | 8 |
| `alpaca_eval` | length_controlled_winrate=5.981 | n=805 (✓ 0, ✗ 0, ? 805) | **all** | 805 | 152.2 | 466.0 | 46 | 285 |
| ↳ correct | | | **✓** | 0 | – | – | – | – |
| ↳ incorrect | | | **✗** | 0 | – | – | – | – |
| `ifeval_mt_wildchat_unused_withRewrite` | prompt_level_loose_acc=0.6979 | n=1774 (✓ 1238, ✗ 536, ? 0) | **all** | 1774 | 54.5 | 368.3 | 6 | 28 |
| ↳ correct | | | **✓** | 1238 | 31.8 | 278.8 | 6 | 24 |
| ↳ incorrect | | | **✗** | 536 | 107.0 | 515.2 | 7 | 114 |
| `ifeval_mt_ood_wildchat_unused_withRewrite` | prompt_level_loose_acc=0.5559 | n=1387 (✓ 771, ✗ 616, ? 0) | **all** | 1387 | 176.2 | 796.4 | 12 | 299 |
| ↳ correct | | | **✓** | 771 | 85.7 | 569.2 | 4 | 84 |
| ↳ incorrect | | | **✗** | 616 | 289.4 | 999.7 | 23 | 486 |
| `ifeval_ood` | prompt_level_loose_acc=0.4033 | n=300 (✓ 121, ✗ 179, ? 0) | **all** | 300 | 123.7 | 681.7 | 3 | 56 |
| ↳ correct | | | **✓** | 121 | 11.7 | 31.5 | 2 | 23 |
| ↳ incorrect | | | **✗** | 179 | 199.4 | 874.1 | 3 | 104 |
| `livecodebench_codegeneration` | pass_at_1=0.03595 | n=612 (✓ 22, ✗ 590, ? 0) | **all** | 612 | 25.4 | 27.6 | 10 | 63 |
| ↳ correct | | | **✓** | 22 | 49.2 | 22.7 | 56 | 80 |
| ↳ incorrect | | | **✗** | 590 | 24.6 | 27.4 | 10 | 59 |
| `minerva_math_500` | exact_match_flex=0.236 | n=500 (✓ 118, ✗ 382, ? 0) | **all** | 500 | 7.8 | 3.6 | 7 | 10 |
| ↳ correct | | | **✓** | 118 | 7.9 | 3.9 | 7 | 10 |
| ↳ incorrect | | | **✗** | 382 | 7.7 | 3.5 | 7 | 10 |

_Tok columns are TOKEN counts across all model_output samples (pass@k contributes k samples per item). Stratified into all / correct items / incorrect items based on the per-item primary metric; ✓ rows are samples from items where the primary metric > 0, ✗ rows are samples from items where it equals 0. '? items' are items whose per-item metrics don't expose the primary metric (e.g. alpaca's length_controlled_winrate is aggregate-only). Full distributions in per-subtask `*-length_stats.json`._

<!-- END: summarize.py output -->

### Interpretation

The model **reward-hacked the shaping objective**. The picture in tokens is
even starker than in characters: on AIME the entire 960-sample distribution
sits at **mean 7.4 tokens, std 0.8** — the model converged to emitting exactly
one `\boxed{X}`-style token sequence with essentially no variance. Minerva is
similar (mean 7.8 tokens) and Minerva's correct vs. incorrect distributions
are statistically indistinguishable (✓ 7.9 vs. ✗ 7.7), which says the model
isn't even reasoning longer when it has a harder problem — it just always
emits one short guess.

This is the predicted-but-undesired failure mode (Key Challenges section of the
proposal: *"Condensed reasoning may be ineffective: shorter reasoning chains
induced by our method may be strictly less effective than unbounded reasoning"*).
With α=1.0 and `constant` warm-up applied from step 0, the model never gets to
learn that chain-of-thought helps before length pressure pushes it toward terse
outputs. Within each prompt group, the shortest correct response sets L_min, and
α=1.0 zeroes out any response 2× L_min or longer — so the global optimum is to
make L_min as small as possible.

#### Where the correct-vs-incorrect length split is informative

- **livecodebench**: correct samples are noticeably *longer* (mean 49.2 tok vs.
  incorrect 24.6) — code can't be functional in a single token, so the few
  passes survived precisely because the model emitted enough code. This is the
  cleanest signal that length pressure damaged the model's ability to use
  enough tokens to solve the task.
- **ifbench (all three subtasks)**: correct samples are *shorter* than incorrect
  (e.g. ifeval_ood ✓ 11.7 vs. ✗ 199.4). Instruction-following items that
  required terse outputs got nailed; ones that needed longer responses got
  truncated to gibberish. Same length-collapse story, just routed through a
  different task structure.
- **AIME / Minerva**: there's nothing to split on AIME (0 correct items).
  Minerva ✓ and ✗ are indistinguishable in length (~7.8 tok either way),
  reinforcing that the model is guessing rather than reasoning.

#### Asymmetric off-distribution check

On **AlpacaEval** (absent from the training mix, so no length pressure was
applied) the model still produces ordinary multi-hundred-character responses
(mean 152 tokens, p90 285). The collapse is confined to the verifiable-reward
distribution — strong evidence this is a learned strategy specific to where the
reward shaping was applied, not a global capability loss.

#### Training-time curves: `val/scores_pre_shaping` vs `val/scores_post_shaping`

Reading the two curves together from W&B (run `2pkl9fhp`, 1000 steps):

- **`val/scores_pre_shaping`** (the raw verifier reward) stays roughly flat
  around 4 throughout training. No meaningful upward trend — the model's true
  solve rate doesn't improve over training.
- **`val/scores_post_shaping`** (the reward the loss actually trains against)
  starts much lower at ~2, climbs steadily, and **converges up to the pre-
  shaping level by ~step 400**. From step 400 onward, post ≈ pre.

The convergence of post to pre — not a widening gap as one might initially
predict — is the smoking-gun story. The mechanism:

1. Early training: many correct rollouts of varying lengths exist; shaping
   zeros out the longer ones; post-shaping mean sits well below pre.
2. The model adapts not by *solving more* but by *making its existing correct
   rollouts shorter*, so fewer get zeroed.
3. By ~step 400 the gap closes — almost every correct rollout is at-or-near
   L_min and escapes the shaping penalty.
4. The remaining ~600 steps of training make no progress on raw correctness
   (pre_shaping curve is flat). Gradient capacity is entirely spent on
   shortening, not on improving.

Concretely: the model's training-time reward ("post") goes up, but its
training-time correctness ("pre") doesn't. The shape of the gap, not its size,
is the diagnostic.

##### Related: `unsolved_batch_size_ratio` doesn't show progressive collapse

The unsolved-batch ratio falls from ~0.95 → ~0.75 over steps 0–~350 and then
plateaus with noise. It does *not* climb back up — i.e. the failure mode is
NOT "shaping makes more groups have zero correct as training progresses."
Caveat: this metric is computed on **post-shaping** scores, so longer-correct
rollouts that get zeroed are counted as unsolved. The true unsolved rate is
presumably lower. See `design_followups.md` for the reporting-bug fix queued
for the next run.

### What this means for the writeup and next runs

1. **Publishable negative finding.** Aggressive linear shaping on an RL-Zero
   base model collapses reasoning. The Pareto plot will show "shaping makes
   responses dramatically shorter but accuracy on hard reasoning drops to 0."
2. **Need the baseline numbers** (now in flight) to make the contrast rigorous —
   if Jacob's unshaped Qwen baseline retains reasoning and gets non-zero AIME,
   that's exactly the contrast the paper is built on.
3. **Followups suggested by this result**:
   - `WARMUP_TYPE=solve_rate` (threshold 0.3) — only apply length pressure once
     the model can solve some problems, so reasoning has a chance to develop.
   - `DECAY_PARAM=0.5` — gentler slope; longer-but-correct responses still keep
     half their reward.
   - Possibly `binary_shortest` ablation to test the extreme version of what
     just happened.

### Smoke-test caveat (still relevant)

The smoke test (`01KQT5BN2F2HKQ9QH5AN77YRKG`) verified the shaping code path on
Qwen2.5-0.5B-Instruct: scores_pre/post diverged correctly, advantages stayed
bounded, all training steps completed. The collapse here is not a code bug — it
is the shaping doing exactly what it was designed to do, just too aggressively
for this base model.

### What to watch in W&B for the next run

- `val/scores_pre_shaping` vs `val/scores_post_shaping` — divergence should
  grow more slowly with `solve_rate` warm-up (zero until threshold crossed).
- `val/sequence_lengths_solved` over training steps — primary length signal.
  This run's curve from wandb is the smoking gun; we expect a noticeably less
  aggressive drop in the next run.
- `val/sequence_lengths_unsolved` — should be unaffected (shaping does not
  touch incorrect responses). Useful sanity check.
- `val/advantages_min` / `val/advantages_max` — exploding values would
  indicate the shaped reward distribution is destabilizing.

## Known issues

### Attempt 1 — checkpoint_state_dir validation crash (2026-05-04)

`grpo_utils.ExperimentConfig.__post_init__` rejected the config:

```
ValueError: `checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!
```

Root cause: PR #1600 (April 2026) changed `checkpoint_state_freq` default from
`-1` (off) to `200` (on). mason.py auto-injects `--checkpoint_state_dir` *only*
inside the dataset-caching block (`if not skip_caching:` at mason.py:406). Our
launch passes `--no_auto_dataset_cache` (required on macOS where `vllm` isn't
installed), which skips that block entirely, so `checkpoint_state_dir` was
never injected. Validation failed before any training started.

**Fix** (committed): `cse-579-scripts/length_shaping_rl_qwen.sh` and
`length_shaping_rl_7b.sh` now compute a unique `CHECKPOINT_STATE_DIR` and pass
it as `--checkpoint_state_dir` explicitly. Same fix would be needed in
Jacob's `baseline_rl.sh` / `baseline_rl_7b.sh` if launched with
`--no_auto_dataset_cache`.

### Attempt 2 — HF push 403 at end of training (2026-05-04)

After 1000 training steps completed and checkpoints saved, the final
`push_folder_to_hub(allenai/open_instruct_dev, ...)` returned 403 Forbidden
because Ian's HF token (from `ai2/ianm` workspace) does not have write access
to the `allenai` HF organization. Training succeeded; only the upload failed.

**Workaround for next run**: pass `--push_to_hub false` in the launch script,
or set `--hf_entity ianmagnusson` to push to a personal HF repo. We don't
actually need the HF push — checkpoints on weka are sufficient for eval.

### Attempt 2 — auto-launched evals failed (tokenizer compat) (2026-05-04)

The auto-launched eval Beaker jobs (steps 800/900/1000 × 5 tasks) all failed
with:

```
AttributeError: 'list' object has no attribute 'keys'
  in transformers/tokenization_utils_base.py:1182
       _set_model_specific_special_tokens
```

The eval image's `transformers` version expects `extra_special_tokens` as a
dict but the saved Qwen tokenizer has it as a list. `--use_hf_tokenizer_template`
was passed but doesn't bypass this codepath.

**Workaround** (Jacob's): bypass the saved tokenizer entirely by passing
`--tokenizer_path /weka/oe-adapt-default/jacobm/repos/cse-579/tokenizers/qwen3-olmo-thinker-eos-old-transformers`
to the eval submitter. Manual eval submission via
`cse-579-scripts/submit_lenshape_qwen_eval_jobs.sh` (added 2026-05-11)
applies this and was used for the eval jobs listed under Outputs.
