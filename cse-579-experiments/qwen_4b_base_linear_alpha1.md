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

- **Compare to**: Jacob's `cse-579-scripts/baseline_rl.sh` run — no shaping,
  identical otherwise.
- **Baseline checkpoint** (confirmed via WEKA S3 endpoint):
  `/weka/oe-adapt-default/allennlp/deletable_checkpoint/jacobm/baseline_think_run_4b_base_mixed_32k__1__1776217615_checkpoints/step_1000`
  (the `_32k_` variant; matches our pack_length=32768 / response_length=30720).
- **Baseline evals** submitted 2026-05-11 via
  `cse-579-scripts/submit_qwen_baseline_eval_jobs.sh`:
  - alpaca_eval_v3: [01KRCBPQFRK0NRX2238AQDS5EX](https://beaker.org/ex/01KRCBPQFRK0NRX2238AQDS5EX)
  - minerva_math_500: [01KRCBPR99FNA17D8R78B4VCPA](https://beaker.org/ex/01KRCBPR99FNA17D8R78B4VCPA)
  - ifbench::tulu: [01KRCBPS5YSMTJ1H725Z8DF3RN](https://beaker.org/ex/01KRCBPS5YSMTJ1H725Z8DF3RN)
  - livecodebench: [01KRCBPSZ57HVDSQTQZ6HN0835](https://beaker.org/ex/01KRCBPSZ57HVDSQTQZ6HN0835)
  - aime 2025 pass@32: [01KRCBPTRBBMP46EDW65GJ7JPK](https://beaker.org/ex/01KRCBPTRBBMP46EDW65GJ7JPK)
- Baseline experiment will get its own `.md` once evals complete and results are
  fetched into `cse-579-experiments/results/baseline_think_run_4b_base_mixed_32k_step_1000/`.

## Results (step_1000)

Numbers below are pulled programmatically from the saved metrics in
`results/<run_dir>/<task>/`. Regenerate with:

```
uv run python cse-579-experiments/summarize.py lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000
```

(The table is regenerated from disk; don't edit the numbers by hand.)

<!-- BEGIN: summarize.py output for this run; do not edit manually -->

### `lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000`

| Task | Primary score | n | Resp len mean | median | p90 |
|------|---------------|---|---------------|--------|-----|
| aime | aime:zs_cot_r1::pass_at_32_2025_deepseek: 0.0 | 30 | 14 | 11 | 19 |
| alpaca_eval | alpaca_eval_v3::hamish_zs_reasoning_deepseek: 5.98065 | 805 | 707 | 199 | 1272 |
| ifbench-tulu | ifeval_mt_wildchat_unused_withRewrite::tulu: 0.697858<br>ifeval_mt_ood_wildchat_unused_withRewrite::tulu: 0.555876<br>ifeval_ood::tulu: 0.403333 | 3461 | 664 | 31 | 1400 |
| livecodebench_codegeneration | livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite: 0.0359477 | 612 | 91 | 30 | 234 |
| minerva_math_500 | minerva_math_500::hamish_zs_reasoning: 0.236 | 500 | 13 | 10 | 20 |

_Response lengths are character counts of model continuations. Computed by `fetch_eval_results.sh` and saved per-task in `length_stats.json`._

<!-- END: summarize.py output -->

### Interpretation

The model **reward-hacked the shaping objective**. On every task in the training
distribution (aime, minerva_math_500, livecodebench), median response length is
under 30 characters — the model just emits `\boxed{ANSWER}` (or analogous) with
no reasoning. Examples:

- **AIME**: `\boxed{10}` (gold was `70`) — median 11 chars, max 19
- **Minerva**: `\boxed{(3,\frac{\pi}{2}}` — median 10 chars, max 102

This is the predicted-but-undesired failure mode (Key Challenges section of the
proposal: *"Condensed reasoning may be ineffective: shorter reasoning chains
induced by our method may be strictly less effective than unbounded reasoning"*).
With α=1.0 and `constant` warm-up applied from step 0, the model never gets to
learn that chain-of-thought helps before length pressure pushes it toward terse
outputs. Within each prompt group, the shortest correct response sets L_min, and
α=1.0 zeroes out any response 2× L_min or longer — so the global optimum is to
make L_min as small as possible.

Asymmetric evidence that this is a learned strategy rather than a global
capability loss: on **AlpacaEval** (a task absent from the training mix, so no
length pressure was applied) the model still produces ordinary 700-char
responses. The collapse is confined to the verifiable-reward distribution.

Minerva at 23.6% with zero reasoning suggests the model learned to *guess*
numerical answers correctly when the space is small. AIME at 0% confirms this
strategy collapses when the answer space is wide and reasoning is required.

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
