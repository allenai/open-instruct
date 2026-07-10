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

## p=0.75 replicate seeds and p=0.875 sweep (n=8, k=16)

`2k_ngu075_dapo_n8_k16` (above, seed 1 / default) gets 2 more seeds. Also
adding 3 seeds of `p = 0.875` (between 0.75 and 0.9) to fill out the top end
of the sweep.

| Name | never_give_up | Seed | Beaker |
| --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_seed2` | 0.75 | 2 | [01KWFA2CZNTDNZ4G9GFZW44GDD](https://beaker.org/ex/01KWFA2CZNTDNZ4G9GFZW44GDD) |
| `2k_ngu075_dapo_n8_k16_seed3` | 0.75 | 3 | [01KWFA2GQ7K2S9M6HQDBA9WJKR](https://beaker.org/ex/01KWFA2GQ7K2S9M6HQDBA9WJKR) |
| `2k_ngu0875_dapo_n8_k16_seed1` | 0.875 | 1 | [01KWFA2MM1NRRQQ8ZNZ6ECVP7Y](https://beaker.org/ex/01KWFA2MM1NRRQQ8ZNZ6ECVP7Y) |
| `2k_ngu0875_dapo_n8_k16_seed2` | 0.875 | 2 | [01KWFA2RGJZ1K6WDHRX4KZF7Q4](https://beaker.org/ex/01KWFA2RGJZ1K6WDHRX4KZF7Q4) |
| `2k_ngu0875_dapo_n8_k16_seed3` | 0.875 | 3 | [01KWFA2W47N6VP0CRV1SQHRVV7](https://beaker.org/ex/01KWFA2W47N6VP0CRV1SQHRVV7) |

### Launch commands

```bash
OC=true EXP=2k_ngu075_dapo_n8_k16_seed2 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.75 --seed 2

OC=true EXP=2k_ngu075_dapo_n8_k16_seed3 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.75 --seed 3

OC=true EXP=2k_ngu0875_dapo_n8_k16_seed1 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.875 --seed 1

OC=true EXP=2k_ngu0875_dapo_n8_k16_seed2 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.875 --seed 2

OC=true EXP=2k_ngu0875_dapo_n8_k16_seed3 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.875 --seed 3
```

## Relaunch of stalled/crashed runs (post eval-callback fix, 59ba75962)

Three replicate/sweep runs stalled (spinning in `accumulate_inference_batches`,
never reaching the target prompt count) and one (`p=0.875` seed2) crashed on
an NCCL `ALLGATHER` collective timeout (rank desync, SIGABRT). Killed the
stalled Beaker jobs and relaunched all four against the same
`michaeln/open-instruct-integration-test-ngu` image, rebuilt for commit
`59ba75962` ("Gate OLMo-core eval collection on pending rounds").

| Name | never_give_up | Seed | Prior Beaker (stalled/crashed) | New Beaker |
| --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_seed2` | 0.5 | 2 | [01KWDYT89WTRYXNJ18CCCBW6QY](https://beaker.org/ex/01KWDYT89WTRYXNJ18CCCBW6QY) (stalled) | [01KWK54TKZ8ZEJHCZ89RV4SGZ3](https://beaker.org/ex/01KWK54TKZ8ZEJHCZ89RV4SGZ3) |
| `2k_ngu075_dapo_n8_k16_seed3` | 0.75 | 3 | [01KWFA2GQ7K2S9M6HQDBA9WJKR](https://beaker.org/ex/01KWFA2GQ7K2S9M6HQDBA9WJKR) (stalled) | [01KWK5602BSM5MW1VPTS198FC0](https://beaker.org/ex/01KWK5602BSM5MW1VPTS198FC0) |
| `2k_baseline_dapo_n8_k16_seed3` | — | 3 | [01KWDYTXAD4D7SXP5MMAE69VVB](https://beaker.org/ex/01KWDYTXAD4D7SXP5MMAE69VVB) (stalled) | [01KWK57GZ3TNVA1WPVXX4H482S](https://beaker.org/ex/01KWK57GZ3TNVA1WPVXX4H482S) |
| `2k_ngu0875_dapo_n8_k16_seed2` | 0.875 | 2 | [01KWFA2RGJZ1K6WDHRX4KZF7Q4](https://beaker.org/ex/01KWFA2RGJZ1K6WDHRX4KZF7Q4) (crashed) | [01KWK57SNVNP9EMJKHC3R6JRDT](https://beaker.org/ex/01KWK57SNVNP9EMJKHC3R6JRDT) |

Checked the other long-running unfinalized jobs: `2k_ngu0875_dapo_n8_k16_seed{1,3}`
are progressing normally (training_step advancing steadily), just slow — left
alone. `2k_baseline_dapo_n4_k32_seed2` was genuinely stalled too (stuck at
`training_step=1896`, same all-zero-reward-filtering spin as the others).
Killed it and relaunched resuming from its existing checkpoint state rather
than from scratch, by passing the original `--checkpoint_state_dir` (mason.py
only auto-replaces that path if it isn't already under `/weka/`):

| Name | Prior Beaker (stalled) | New Beaker (resumed) |
| --- | --- | --- |
| `2k_baseline_dapo_n4_k32_seed2` | [01KWDYV1187B947VMM72YRD1W2](https://beaker.org/ex/01KWDYV1187B947VMM72YRD1W2) | [01KWK5Z2DR49G5XGDDT8DVWRD4](https://beaker.org/ex/01KWK5Z2DR49G5XGDDT8DVWRD4) |

Confirmed resume via logs: `[DataPreparationActor] Restored state:
training_step=1800, last_consumed_step=1799` (the last checkpoint before the
stall at step 1896, `checkpoint_state_freq=100`).

```bash
OC=true EXP=2k_baseline_dapo_n4_k32_seed2 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 --max_grad_norm 5.0 --seed 2 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1782879978_178800
```

### Launch commands (repeat per config/seed)

```bash
OC=true EXP=2k_ngu05_dapo_n8_k16_seed2 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.5 --seed 2

OC=true EXP=2k_ngu075_dapo_n8_k16_seed3 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.75 --seed 3

OC=true EXP=2k_baseline_dapo_n8_k16_seed3 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --seed 3

OC=true EXP=2k_ngu0875_dapo_n8_k16_seed2 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.875 --seed 2
```

## Seed 4 for p=0.5 and p=0.75 (n=8, k=16)

Both `p = 0.5` and `p = 0.75` already had seeds 1–3, so this adds a 4th seed
to each, against the same `michaeln/open-instruct-integration-test-ngu` image
(still current as of `6667e6ea5`; no code changes since the `59ba75962`
rebuild).

| Name | never_give_up | Seed | Beaker |
| --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_seed4` | 0.5 | 4 | [01KX48X5PRGBKCHEABHA393W3N](https://beaker.org/ex/01KX48X5PRGBKCHEABHA393W3N) |
| `2k_ngu075_dapo_n8_k16_seed4` | 0.75 | 4 | [01KX48XFQRW22M1F06QECM80H0](https://beaker.org/ex/01KX48XFQRW22M1F06QECM80H0) |

### Launch commands

```bash
OC=true EXP=2k_ngu05_dapo_n8_k16_seed4 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.5 --seed 4

OC=true EXP=2k_ngu075_dapo_n8_k16_seed4 BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 --max_grad_norm 5.0 --never_give_up 0.75 --seed 4
```

## Additional seeds with grad norm 1.0

One new seed for every baseline except k=8, plus one new n=8/k=16 NGU seed at
each of `p = 0.5`, `p = 0.75`, and `p = 0.875`. Unlike the earlier runs, all
six use `--max_grad_norm 1.0`.

| Name | n | k | never_give_up | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | — | 1 | [01KX6VGB98S6MRT5FWY2WTFYJ1](https://beaker.org/ex/01KX6VGB98S6MRT5FWY2WTFYJ1) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed1` | 4 | 32 | — | 1 | [01KX6VGVE15SX40F089KZB3VM0](https://beaker.org/ex/01KX6VGVE15SX40F089KZB3VM0) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed1` | 2 | 64 | — | 1 | [01KX6VH09F9KZ97HNM6KJ548RX](https://beaker.org/ex/01KX6VH09F9KZ97HNM6KJ548RX) |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.5 | 1 | [01KX6VHGEMJBGGGX3DNKGGA4TV](https://beaker.org/ex/01KX6VHGEMJBGGGX3DNKGGA4TV) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.75 | 1 | [01KX6VJ1BF00FZXBSDH5QX6ADG](https://beaker.org/ex/01KX6VJ1BF00FZXBSDH5QX6ADG) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.875 | 1 | [01KX6VJ7FN7DQHHWWVWF91174W](https://beaker.org/ex/01KX6VJ7FN7DQHHWWVWF91174W) |

### Launch command (repeat per row)

```bash
OC=true EXP=$NAME \
  ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout $N --num_samples_per_prompt_rollout $K \
  --max_grad_norm 1.0 --seed $SEED [--never_give_up $P]
```

## Best-step held-out evals (BRUMO / HMMT / AIME 2025)

Eval-only runs (`open_instruct/grpo.py --eval_only`, commit `23fcabfe5`) of every
included seed at its best in-training AIME `eval/pass_at_1` step, on
`mnoukhov/brumo_2025_openinstruct`, `mnoukhov/hmmt_feb_2025_openinstruct`,
`mnoukhov/hmmt_nov_2025_openinstruct`, and `mnoukhov/aime_2025_openinstruct`
(prompt/answer-identical to the training eval's `allenai/aime_2025_openinstruct`,
but with a distinct `dataset` label). Sampling matches the in-training AIME evals
(pass@32, temperature 1.0, eval_top_p 0.95, 8192-token responses,
`qwen_instruct_user_boxed_math`); per-competition scores land in
`eval/pass_at_1/<label>` etc., wandb group `deepscaler_eval_best`.

Native OLMo-core checkpoint states were converted to HF via
`scripts/train/convert_olmo_core_to_hf.py` (fixed to use OLMo-core's DCP loader;
conversion verified bit-exact against the step2000 HF export of
`baseline_dapo_n16_k8` seed 1) into
`/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/best_aime_hf/`.
Smoke test (Qwen3-0.6B-Base, pass@2, 1k tokens):
[01KX4B0RSSC259B0GQB1N2GNE9](https://beaker.org/ex/01KX4B0RSSC259B0GQB1N2GNE9).

Launched via `scripts/train/qwen/qwen3_4b_deepscaler_eval_best_all.sh` with the
`michaeln/open-instruct-integration-test-ngu` image. Four first-attempt jobs died
at Ray head startup (transient Redis session-name mismatch) and were relaunched;
the table links point at the successful attempts. The `ngu0875_dapo_n8_k16` seed 1
job exited 134 during post-eval teardown but had already completed the eval and
logged all metrics (wandb `rjyxvuxy`), so it was not rerun. The failures happen
when Beaker packs two of these 4-GPU jobs onto one node: `ray_node_setup.sh`
hardcodes the Ray head port on host networking, so the second job joins the
first job's Ray cluster and dies. The `ngu05` seed 2 and `ngu075` seed 4 evals
were relaunched as full-node `NUM_GPUS=8` jobs to avoid packing.

| Config | Seed | Best AIME step | Source wandb | Eval Beaker |
| --- | --- | --- | --- | --- |
| `baseline_dapo_n16_k8` | 1 | 1100 | ys9ymj6v | [01KX4CJ007144ERATR0YQ39560](https://beaker.org/ex/01KX4CJ007144ERATR0YQ39560) |
| `baseline_dapo_n16_k8` | 2 | 1100 | w24n9eea | [01KX4CP47AM3S3GQAZJ54QRZ1A](https://beaker.org/ex/01KX4CP47AM3S3GQAZJ54QRZ1A) |
| `baseline_dapo_n16_k8` | 3 | 1900 | xpmxgh19 | [01KX4CTSG18B94WVBVJE21SAXT](https://beaker.org/ex/01KX4CTSG18B94WVBVJE21SAXT) |
| `baseline_dapo_n8_k16` | 1 | 900 | cmoi4l53 | [01KX4CY838RPWJ9YJ1755DZGZ0](https://beaker.org/ex/01KX4CY838RPWJ9YJ1755DZGZ0) |
| `baseline_dapo_n8_k16` | 2 | 1400 | v12nc9i8 | [01KX4D1TSG0EQ8T4F45A1M150A](https://beaker.org/ex/01KX4D1TSG0EQ8T4F45A1M150A) |
| `baseline_dapo_n8_k16` | 3 | 1500 | 02frsm05 | [01KX6B8SB5FB5SRB3XSMCHE3X3](https://beaker.org/ex/01KX6B8SB5FB5SRB3XSMCHE3X3) |
| `baseline_dapo_n4_k32` | 1 | 1200 | t68pmy9s | [01KX6B96SMH1GBC4ZW6V4A6WD0](https://beaker.org/ex/01KX6B96SMH1GBC4ZW6V4A6WD0) |
| `baseline_dapo_n4_k32` | 2 | 800 | midlg5cv+txilsd7o | [01KX4DBJM8GS524VXW6BGRZAP0](https://beaker.org/ex/01KX4DBJM8GS524VXW6BGRZAP0) |
| `baseline_dapo_n4_k32` | 3 | 1200 | z4448u5r | [01KX4DEK3TA05HTZAPQ95ES9ZQ](https://beaker.org/ex/01KX4DEK3TA05HTZAPQ95ES9ZQ) |
| `baseline_dapo_n2_k64` | 1 | 1000 | pxlpna71 | [01KX4DHTHWVEDPV9CM27KR3MM3](https://beaker.org/ex/01KX4DHTHWVEDPV9CM27KR3MM3) |
| `baseline_dapo_n2_k64` | 2 | 1600 | blfze1rc | [01KX4DMW5BYJBSKD2T99VYYBCH](https://beaker.org/ex/01KX4DMW5BYJBSKD2T99VYYBCH) |
| `baseline_dapo_n2_k64` | 3 | 800 | hl3d7uml | [01KX4DRF974YG8JWCWRP1JGD12](https://beaker.org/ex/01KX4DRF974YG8JWCWRP1JGD12) |
| `ngu05_dapo_n8_k16` | 1 | 1600 | 6rxe8lh5 | [01KX6B9KJGHYBSTDWCN1HJJ2R7](https://beaker.org/ex/01KX6B9KJGHYBSTDWCN1HJJ2R7) |
| `ngu05_dapo_n8_k16` | 2 | 1300 | ivrq5tsx | [01KX6BWQQ0S5C24TTMV69XW3N4](https://beaker.org/ex/01KX6BWQQ0S5C24TTMV69XW3N4) |
| `ngu05_dapo_n8_k16` | 3 | 1000 | pdm9oqd6 | [01KX6BA0RD5E5A9BPQPSAAWE6R](https://beaker.org/ex/01KX6BA0RD5E5A9BPQPSAAWE6R) |
| `ngu075_dapo_n8_k16` | 1 | 1000 | i12fv1iu | [01KX4E68AKMJS75B05WCD3ZBFX](https://beaker.org/ex/01KX4E68AKMJS75B05WCD3ZBFX) |
| `ngu075_dapo_n8_k16` | 3 | 1100 | ai4avb1d | [01KX4E9B224W6HDAZGS8564BAB](https://beaker.org/ex/01KX4E9B224W6HDAZGS8564BAB) |
| `ngu075_dapo_n8_k16` | 4 | 900 | kg4ycwi8 | [01KX6BX4K3SQZ7E4F9YM1Y2XAR](https://beaker.org/ex/01KX6BX4K3SQZ7E4F9YM1Y2XAR) |
| `ngu0875_dapo_n8_k16` | 1 | 1700 | x5rkqi9n | [01KX4ECBJNV5NG1PDXTFJ3TWJK](https://beaker.org/ex/01KX4ECBJNV5NG1PDXTFJ3TWJK) |
| `ngu0875_dapo_n8_k16` | 2 | 1700 | ux8zlyun | [01KX4EFPEA1F4WMAMQSKY982H3](https://beaker.org/ex/01KX4EFPEA1F4WMAMQSKY982H3) |
| `ngu0875_dapo_n8_k16` | 3 | 1000 | 0f6tb0za | [01KX4EK2WFQPKYQBFWCECN5YM5](https://beaker.org/ex/01KX4EK2WFQPKYQBFWCECN5YM5) |

## Smoke test (2 GPU, before launching the sweep)

Quick NGU + per-quartile-metrics check on a small model via
`scripts/train/debug/ngu_quartiles_2gpu.sh`.

| Name | Notes | Beaker |
| --- | --- | --- |
| `ngu_quartiles_2gpu` | 2 GPU, Qwen3-0.6B-Base, 256 episodes, `--active_sampling --never_give_up 1.0` | [01KW2XH2WYC158J2ESK4S1F3TY](https://beaker.org/ex/01KW2XH2WYC158J2ESK4S1F3TY) |
