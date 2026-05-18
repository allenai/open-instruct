# Qwen3-4B-Base RL-Zero · baseline (no shaping)

## Status

- **State**: training completed (Jacob ran this; checkpoints exist on weka)
- **Eval state**: retrieved (5 oe-eval beaker jobs for step_1000)
- **Last updated**: 2026-05-11

## Purpose

The no-shaping comparison point for `qwen_4b_base_linear_alpha1`. Same RLVR
recipe (Qwen3-4B-Base, `jacobmorrison/cse-579-mixed-rl`, same hyperparams),
but no `--length_reward_shaping_method` arg → standard GRPO advantages with no
length-aware adjustments. Used to attribute behavior in the shaping run to the
shaping itself rather than to base-model RL-Zero in general.

## Beaker

- **Training run**: launched by Jacob from `cse-579-scripts/baseline_rl.sh` /
  `baseline_rl_32k.sh`. Author = `jacobm`. Training Beaker IDs not retained in
  the eval script; checkpoint timestamps on weka show 2026-04-15.
- **Checkpoint** (the one we evaluate here):
  `/weka/oe-adapt-default/allennlp/deletable_checkpoint/jacobm/baseline_think_run_4b_base_mixed_32k__1__1776217615_checkpoints/step_1000`
  — confirmed via WEKA S3 endpoint. Has `.checkpoint_complete` marker, 8.0 GB
  safetensors, full tokenizer/config.

## Configuration

- **Launch script**: `cse-579-scripts/baseline_rl.sh` (Jacob's, the `_32k`
  variant; matches our shaping run's pack_length=32768 / response_length=30720)
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl`
- **Shaping**: none (`length_reward_shaping_method=none`, default)
- **All other hyperparams**: identical to the linear-α=1.0 run

## Outputs

- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/jacobm/baseline_think_run_4b_base_mixed_32k__1__1776217615_checkpoints/`
- **Eval beaker jobs** (submitted via `cse-579-scripts/submit_qwen_baseline_eval_jobs.sh` on 2026-05-11):
  - alpaca_eval_v3: [01KRCBPQFRK0NRX2238AQDS5EX](https://beaker.org/ex/01KRCBPQFRK0NRX2238AQDS5EX)
  - minerva_math_500: [01KRCBPR99FNA17D8R78B4VCPA](https://beaker.org/ex/01KRCBPR99FNA17D8R78B4VCPA)
  - ifbench::tulu: [01KRCBPS5YSMTJ1H725Z8DF3RN](https://beaker.org/ex/01KRCBPS5YSMTJ1H725Z8DF3RN)
  - livecodebench: [01KRCBPSZ57HVDSQTQZ6HN0835](https://beaker.org/ex/01KRCBPSZ57HVDSQTQZ6HN0835)
  - aime 2025 pass@32: [01KRCBPTRBBMP46EDW65GJ7JPK](https://beaker.org/ex/01KRCBPTRBBMP46EDW65GJ7JPK) (exit 143; metrics were written before SIGTERM during datalake upload — see Known issues)
- **Eval results path**: `cse-579-experiments/results/baseline_think_run_4b_base_mixed_32k/step_1000/`. Regenerate the table below with `uv run python cse-579-experiments/summarize.py baseline_think_run_4b_base_mixed_32k/step_1000`.

## Results (step_1000)

<!-- BEGIN: summarize.py output for this run; do not edit manually -->

### `baseline_think_run_4b_base_mixed_32k/step_1000`

| Task | Primary | Items (✓/✗/?) | Subset | gens | Tok mean | Tok std | Tok p50 | Tok p90 |
|------|---------|----------------|--------|------|----------|---------|---------|---------|
| `aime` | pass_at_1=0.0125 | n=30 (✓ 5, ✗ 25, ? 0) | **all** | 960 | 1840.4 | 2152.6 | 564 | 5422 |
| ↳ correct | | | **✓** | 160 | 1600.3 | 1935.5 | 624 | 5519 |
| ↳ incorrect | | | **✗** | 800 | 1888.4 | 2190.3 | 536 | 5404 |
| `alpaca_eval` | length_controlled_winrate=6.559 | n=805 (✓ 0, ✗ 0, ? 805) | **all** | 805 | 1067.0 | 1708.5 | 325 | 4238 |
| ↳ correct | | | **✓** | 0 | – | – | – | – |
| ↳ incorrect | | | **✗** | 0 | – | – | – | – |
| `ifeval_mt_wildchat_unused_withRewrite` | prompt_level_loose_acc=0.7334 | n=1774 (✓ 1301, ✗ 473, ? 0) | **all** | 1774 | 423.8 | 1207.9 | 96 | 523 |
| ↳ correct | | | **✓** | 1301 | 279.3 | 866.0 | 90 | 440 |
| ↳ incorrect | | | **✗** | 473 | 821.3 | 1787.2 | 107 | 4207 |
| `ifeval_mt_ood_wildchat_unused_withRewrite` | prompt_level_loose_acc=0.5566 | n=1387 (✓ 772, ✗ 615, ? 0) | **all** | 1387 | 768.3 | 1714.4 | 103 | 4119 |
| ↳ correct | | | **✓** | 772 | 396.5 | 1205.7 | 85 | 481 |
| ↳ incorrect | | | **✗** | 615 | 1235.0 | 2100.6 | 130 | 4890 |
| `ifeval_ood` | prompt_level_loose_acc=0.4367 | n=300 (✓ 131, ✗ 169, ? 0) | **all** | 300 | 717.5 | 1637.1 | 64 | 4408 |
| ↳ correct | | | **✓** | 131 | 470.1 | 1343.3 | 44 | 503 |
| ↳ incorrect | | | **✗** | 169 | 909.2 | 1809.6 | 89 | 4768 |
| `livecodebench_codegeneration` | pass_at_1=0.0768 | n=612 (✓ 47, ✗ 565, ? 0) | **all** | 612 | 174.2 | 377.3 | 126 | 267 |
| ↳ correct | | | **✓** | 47 | 73.5 | 36.2 | 64 | 137 |
| ↳ incorrect | | | **✗** | 565 | 182.6 | 391.3 | 130 | 272 |
| `minerva_math_500` | exact_match_flex=0.542 | n=500 (✓ 271, ✗ 229, ? 0) | **all** | 500 | 1157.5 | 1666.6 | 440 | 4917 |
| ↳ correct | | | **✓** | 271 | 429.6 | 322.0 | 359 | 787 |
| ↳ incorrect | | | **✗** | 229 | 2019.0 | 2138.3 | 608 | 5146 |

<!-- END: summarize.py output -->

## Pair / treatment

- **Compare to**: [`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md)
  — same run with linear α=1.0 length shaping applied from step 0.

### Headline comparison (baseline → treatment)

| Task | Baseline acc | Treatment acc | Δ acc | Baseline tok mean | Treatment tok mean | Length compression |
|------|-------------|--------------|-------|-------------------|-------------------|---------------------|
| AIME (pass_at_1) | 1.25% | 0.0% | −1.25pp | 1840 | 7 | **263×** |
| AIME (pass_at_32) | 16.7% | 0.0% | −16.7pp | — | — | — |
| Minerva (exact_match_flex) | 54.2% | 23.6% | **−30.6pp** | 1158 | 7.8 | **148×** |
| LiveCodeBench (pass_at_1) | 7.7% | 3.6% | −4.1pp | 174 | 25 | 7× |
| ifbench wildchat | 73.3% | 69.8% | −3.5pp | 424 | 54 | 8× |
| ifbench wildchat OOD | 55.7% | 55.6% | ~0 | 768 | 176 | 4× |
| ifbench OOD | 43.7% | 40.3% | −3.4pp | 718 | 124 | 6× |
| AlpacaEval (LC winrate) | 6.56 | 5.98 | −0.6 | 1067 | 152 | 7× |

### Interpretation

- **Hard reasoning collapsed.** Minerva math dropped **30.6 percentage points**
  while average response shrank **148×** (1158 → 8 tokens). AIME pass@32 went
  16.7% → 0%. These are the verifier-trained, hard-reasoning tasks where
  chain-of-thought matters most.
- **The natural length pattern was destroyed.** Baseline minerva shows
  ✓ 430 tok vs ✗ 2019 tok — the model spontaneously reasoned longer on harder
  problems. Treatment shows ✓ 7.9 vs ✗ 7.7 — indistinguishable. The model
  stopped adjusting reasoning length to difficulty.
- **Instruction following mostly survived.** ifbench (all 3 subtasks) shows
  ≤3.5pp accuracy drop despite 4–8× length compression. Many ifbench items
  have terse correct answers, so shortening doesn't kill them.
- **Out-of-distribution behavior held.** AlpacaEval is absent from the training
  mix; the model still produces multi-hundred-token responses there (mean 152
  vs baseline 1067), confirming the collapse is a training-distribution artifact
  rather than a global capability loss.
- **The baseline is doing real reasoning.** AIME pass_at_32 16.7% (5/30 items
  with at least one correct sample out of 32) with mean 1840 tokens per
  rollout — that's actual chain-of-thought, not lucky guessing.

## Known issues

### AIME eval: first attempt preempted, retry succeeded (2026-05-11)

The Beaker experiment for AIME baseline had two jobs:
- Job 1 (`01KRCBPV7SP1VGRH4E5KWHTRVT`): preempted at 20:37 (exit 143).
- Job 2 (`01KRCC4FQX6K2YNF1TR22KV15D`): retry, succeeded at 21:05 (exit 0).

Initial fetcher iteration looked only at `.jobs[0]` (the preempted one) and
appeared to skip the experiment. **Fixed in `fetch_eval_results.sh`** to pick
the latest successful job across all retries; the AIME metrics are now fetched
correctly by the unmodified pipeline. Preemption + retry is expected at normal
priority for the bulk intermediate-step evals, so this fix is load-bearing.
