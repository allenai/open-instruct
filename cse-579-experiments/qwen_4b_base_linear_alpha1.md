# Qwen3-4B-Base RL-Zero · linear shaping, α=1.0

## Status

- **State**: training completed (attempt 2; HF push failed but training and checkpoints were fine)
- **Eval state**: in progress (5 oe-eval beaker jobs submitted 2026-05-11 for step_1000)
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
- **Eval results path**: see oe-eval datalake (`run_id=placeholder`) once jobs complete

## Pair / baseline

- **Compare to**: Jacob's `cse-579-scripts/baseline_rl.sh` run — no shaping,
  identical otherwise. **Baseline experiment doc still needs creating once Jacob
  launches it.**

## Notes

- This is the first run that uses the auto-derived `correctness_threshold`
  (sentinel `-1.0` resolved to `0.0` here because format reward is off).
- Smoke test (`01KQT5BN2F2HKQ9QH5AN77YRKG`) verified the shaping code path on
  Qwen2.5-0.5B-Instruct: scores_pre/post diverged correctly, advantages stayed
  bounded, all training steps completed. So we have confidence the integration
  itself isn't going to blow up — what we're testing here is whether the
  shaping at scale produces meaningful length reduction without tanking accuracy.

### What to watch in W&B

- `val/scores_pre_shaping` vs `val/scores_post_shaping` — should diverge as
  solve rate climbs. Early steps (low solve rate) will likely show pre==post
  because each group has at most one correct response.
- `val/sequence_lengths_solved` over training steps — primary length signal.
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
