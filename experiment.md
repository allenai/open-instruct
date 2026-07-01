# Experiments: DeepScaleR difficulty-quartile DAPO baselines

Runs on `mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles` (splits
concatenated into `train`; the per-example `dataset` column is
`math_deepscaler_quartile{0,1,2,3}`, all routed to the `math` verifier via the
`math_*` prefix fallback). Per-quartile batch metrics are logged under
`batch/nonzero_prompts/<quartile>`, `batch/completions_used/<quartile>`,
`batch/filtered_prompts_solved/<quartile>`, etc.

Launched via `scripts/train/qwen/qwen3_4b_deepscaler_math.sh` with `OC=true`
(OLMo-core GRPO / FSDP) against the prebuilt image
`michaeln/open-instruct-integration-test-ngu`.

## Baseline sweep (n × k = 128 completions/step held constant)

`--total_episodes 256000` ⇒ 256000 / 128 = **2000 steps** (the "2k" prefix).
All runs add `--max_grad_norm 5.0`. n = `--num_unique_prompts_rollout`,
k = `--num_samples_per_prompt_rollout`.

| Name | n (prompts) | k (samples) | Status / Beaker |
| --- | --- | --- | --- |
| `2k_baseline_dapo_n16_k8` | 16 | 8 | [01KW2Y5BBT0KAJRPRYD18AEPH0](https://beaker.org/ex/01KW2Y5BBT0KAJRPRYD18AEPH0) |
| `2k_baseline_dapo_n8_k16` | 8 | 16 | [01KW2Y5F9NZJJJ9DGBDYPXAWXV](https://beaker.org/ex/01KW2Y5F9NZJJJ9DGBDYPXAWXV) |
| `2k_baseline_dapo_n4_k32` | 4 | 32 | [01KW2Y5K5X36G079XXGBH4AS59](https://beaker.org/ex/01KW2Y5K5X36G079XXGBH4AS59) |
| `2k_baseline_dapo_n2_k64` | 2 | 64 | [01KW2Y5PSGZ9CFS1D4QMWT8HGF](https://beaker.org/ex/01KW2Y5PSGZ9CFS1D4QMWT8HGF) |

### Launch commands

```bash
OC=true EXP=2k_baseline_dapo_n16_k8 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 --max_grad_norm 5.0

OC=true EXP=2k_baseline_dapo_n8_k16 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0

OC=true EXP=2k_baseline_dapo_n4_k32 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 --max_grad_norm 5.0

OC=true EXP=2k_baseline_dapo_n2_k64 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 2 --num_samples_per_prompt_rollout 64 --max_grad_norm 5.0
```

## NGU sweep (same base, add `--never_give_up`)

Only for the n=16/k=8 and n=8/k=16 configs, at never_give_up 0.5 and 0.9.
Everything else matches the baseline sweep (`--total_episodes 256000`,
`--max_grad_norm 5.0`, `--active_sampling` from the script).

| Name | n | k | never_give_up | Status / Beaker |
| --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n16_k8` | 16 | 8 | 0.5 | [01KW2Y5TEEM0KGZ937HG0KQ7FX](https://beaker.org/ex/01KW2Y5TEEM0KGZ937HG0KQ7FX) |
| `2k_ngu09_dapo_n16_k8` | 16 | 8 | 0.9 | [01KW2Y5Y5WRKHR3F34PEK10XHY](https://beaker.org/ex/01KW2Y5Y5WRKHR3F34PEK10XHY) |
| `2k_ngu05_dapo_n8_k16` | 8 | 16 | 0.5 | [01KW2Y61T14JN8832P1B21J5B7](https://beaker.org/ex/01KW2Y61T14JN8832P1B21J5B7) |
| `2k_ngu09_dapo_n8_k16` | 8 | 16 | 0.9 | [01KW2Y65JEYJR4CN4B4YCNHFFY](https://beaker.org/ex/01KW2Y65JEYJR4CN4B4YCNHFFY) |

### Launch commands

```bash
OC=true EXP=2k_ngu05_dapo_n16_k8 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 --max_grad_norm 5.0 --never_give_up 0.5

OC=true EXP=2k_ngu09_dapo_n16_k8 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 --max_grad_norm 5.0 --never_give_up 0.9

OC=true EXP=2k_ngu05_dapo_n8_k16 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.5

OC=true EXP=2k_ngu09_dapo_n8_k16 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.9
```

## Replication runs (seeds 2 & 3)

`2k_ngu05_dapo_n8_k16` (n=8, k=16, never_give_up=0.5) looked best after the
first sweep, so it and all 4 baselines get 2 more seeds each. All other args
match the original launch for that config; only `--seed` changes.

| Name | Seed | Beaker |
| --- | --- | --- |
| `2k_baseline_dapo_n16_k8_seed2` | 2 | [01KWDYTHDGBE6BKKKQAHWHX49F](https://beaker.org/ex/01KWDYTHDGBE6BKKKQAHWHX49F) |
| `2k_baseline_dapo_n16_k8_seed3` | 3 | [01KWDYTN5YN7PG5DZ34NDCTJ88](https://beaker.org/ex/01KWDYTN5YN7PG5DZ34NDCTJ88) |
| `2k_baseline_dapo_n8_k16_seed2` | 2 | [01KWDYTRVQVY1ZFHHYG7X22PYQ](https://beaker.org/ex/01KWDYTRVQVY1ZFHHYG7X22PYQ) |
| `2k_baseline_dapo_n8_k16_seed3` | 3 | [01KWDYTXAD4D7SXP5MMAE69VVB](https://beaker.org/ex/01KWDYTXAD4D7SXP5MMAE69VVB) |
| `2k_baseline_dapo_n4_k32_seed2` | 2 | [01KWDYV1187B947VMM72YRD1W2](https://beaker.org/ex/01KWDYV1187B947VMM72YRD1W2) |
| `2k_baseline_dapo_n4_k32_seed3` | 3 | [01KWDYV56C7G468K8KQB611KT8](https://beaker.org/ex/01KWDYV56C7G468K8KQB611KT8) |
| `2k_baseline_dapo_n2_k64_seed2` | 2 | [01KWDYV9CSAPMV64W8YBT2HMVY](https://beaker.org/ex/01KWDYV9CSAPMV64W8YBT2HMVY) |
| `2k_baseline_dapo_n2_k64_seed3` | 3 | [01KWDYVCZTQY6KMJRB3CMWZ215](https://beaker.org/ex/01KWDYVCZTQY6KMJRB3CMWZ215) |
| `2k_ngu05_dapo_n8_k16_seed2` | 2 | [01KWDYT89WTRYXNJ18CCCBW6QY](https://beaker.org/ex/01KWDYT89WTRYXNJ18CCCBW6QY) |
| `2k_ngu05_dapo_n8_k16_seed3` | 3 | [01KWDYTD7M6EQ1T6FE9WB04Z55](https://beaker.org/ex/01KWDYTD7M6EQ1T6FE9WB04Z55) |

### Launch command (repeat per config/seed)

```bash
OC=true EXP=${NAME}_seed${SEED} BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout $N --num_samples_per_prompt_rollout $K \
  --max_grad_norm 5.0 --seed $SEED [--never_give_up 0.5]
```

## NGU `p` sweep between 0.5 and 0.9 (n=8, k=16)

One run each at `p = 0.6` and `p = 0.75`, bracketing the best-so-far `p = 0.5`
and the earlier `p = 0.9` run, all at the n=8/k=16 config.

| Name | never_give_up | Beaker |
| --- | --- | --- |
| `2k_ngu06_dapo_n8_k16` | 0.6 | [01KWDYVPNJ5108JDC82YN6HCD6](https://beaker.org/ex/01KWDYVPNJ5108JDC82YN6HCD6) |
| `2k_ngu075_dapo_n8_k16` | 0.75 | [01KWDYVTFRNK4T0JPNBMR2GY1V](https://beaker.org/ex/01KWDYVTFRNK4T0JPNBMR2GY1V) |

### Launch commands

```bash
OC=true EXP=2k_ngu06_dapo_n8_k16 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.6

OC=true EXP=2k_ngu075_dapo_n8_k16 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.75
```

## Smoke test (2 GPU, before launching the sweep)

Quick NGU + per-quartile-metrics check on a small model via
`scripts/train/debug/ngu_quartiles_2gpu.sh`.

| Name | Notes | Beaker |
| --- | --- | --- |
| `ngu_quartiles_2gpu` | 2 GPU, Qwen3-0.6B-Base, 256 episodes, `--active_sampling --never_give_up 1.0` | [01KW2XH2WYC158J2ESK4S1F3TY](https://beaker.org/ex/01KW2XH2WYC158J2ESK4S1F3TY) |
