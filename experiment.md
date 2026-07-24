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
six use `--max_grad_norm 1.0`. The original submissions to `ai2/olmo-instruct`
were canceled and rerun in `ai2/open-instruct-dev`; the table links point to
the replacement runs.

| Name | n | k | never_give_up | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | — | 1 | ~~[01KX6VR8B6EWBFYGH511ATKMXX](https://beaker.org/ex/01KX6VR8B6EWBFYGH511ATKMXX)~~ [01KX6ZB67JSWDTBBH78T4JG0KV](https://beaker.org/ex/01KX6ZB67JSWDTBBH78T4JG0KV) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed1` | 4 | 32 | — | 1 | ~~[01KX6VRNKA2NFV0PJ8AFHEVJZF](https://beaker.org/ex/01KX6VRNKA2NFV0PJ8AFHEVJZF)~~ [01KX6ZC4NV9D7X7HJ1WR8WHS0P](https://beaker.org/ex/01KX6ZC4NV9D7X7HJ1WR8WHS0P) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed1` | 2 | 64 | — | 1 | ~~[01KX6VS3H0MCNPK01R33N5NXWW](https://beaker.org/ex/01KX6VS3H0MCNPK01R33N5NXWW)~~ [01KX6ZCNJ4GVGDHWP141RXET3K](https://beaker.org/ex/01KX6ZCNJ4GVGDHWP141RXET3K) |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed2` | 8 | 16 | — | 2 | [01KXE24SK50VY5RWX0WVBMRPEG](https://beaker.org/ex/01KXE24SK50VY5RWX0WVBMRPEG) |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed3` | 8 | 16 | — | 3 | [01KXE24XX5DMZ573Z2GX378E95](https://beaker.org/ex/01KXE24XX5DMZ573Z2GX378E95) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed2` | 2 | 64 | — | 2 | [01KXE251AT69KM94WWAR77VAQ5](https://beaker.org/ex/01KXE251AT69KM94WWAR77VAQ5) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed3` | 2 | 64 | — | 3 | [01KXE254WXF2QEE7W0EG3VQHV4](https://beaker.org/ex/01KXE254WXF2QEE7W0EG3VQHV4) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed2` | 4 | 32 | — | 2 | [01KXE258GDHF9BSAJ621K68PQF](https://beaker.org/ex/01KXE258GDHF9BSAJ621K68PQF) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed3` | 4 | 32 | — | 3 | [01KXE25C47AH6RH09QSZZ55PG3](https://beaker.org/ex/01KXE25C47AH6RH09QSZZ55PG3) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed4` | 4 | 32 | — | 4 | [01KXE25FQ2AGZ4R8J75C5BDP9C](https://beaker.org/ex/01KXE25FQ2AGZ4R8J75C5BDP9C) |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.5 | 1 | ~~[01KX6VSBVN4E657KCJG301Y36H](https://beaker.org/ex/01KX6VSBVN4E657KCJG301Y36H)~~ [01KX6ZDE9RD8TEVZ2WBTGJFYWM](https://beaker.org/ex/01KX6ZDE9RD8TEVZ2WBTGJFYWM) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.75 | 1 | ~~[01KX6VSKVPWQ27BPJW8859G80G](https://beaker.org/ex/01KX6VSKVPWQ27BPJW8859G80G)~~ [01KX6ZE3HNM57WW32XBQ0K1NA2](https://beaker.org/ex/01KX6ZE3HNM57WW32XBQ0K1NA2) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed1` | 8 | 16 | 0.875 | 1 | ~~[01KX6VSWZ49DB9RH461EZEYZ8R](https://beaker.org/ex/01KX6VSWZ49DB9RH461EZEYZ8R)~~ [01KX6ZENECP864C2SJ2NBDM8KA](https://beaker.org/ex/01KX6ZENECP864C2SJ2NBDM8KA) |

### Launch command (repeat per row)

```bash
OC=true EXP=$NAME \
  ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout $N --num_samples_per_prompt_rollout $K \
  --max_grad_norm 1.0 --seed $SEED [--never_give_up $P]
```

### 2026-07-13: 2 more seeds (n8_k16, n2_k64) + 3 more seeds (n4_k32), all async_steps=4 (default)

n8_k16 and n2_k64 gradnorm1 seed1 ran clean at the default `async_steps=4`,
so just replicated with 2 more seeds each. n4_k32 gradnorm1 seed1 is the one
that collapsed at async4 (see the
[rho_weight-collapse entry](#n4_k32-gradnorm1-stall--rerun-with-async_steps-2))
and was superseded by an async2 rerun — launched 3 more seeds at async4
(rather than 2) specifically to check whether the collapse reproduces across
seeds or was a one-off. Same image (`michaeln/open-instruct-integration-test-ngu`
@ `de6eb6fa6`, no code changes), workspace `ai2/open-instruct-dev`, launched
directly via `bash` (no rebuild needed).

| Name | n | k | Seed | Beaker |
| --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed2` | 8 | 16 | 2 | [01KXE24SK50VY5RWX0WVBMRPEG](https://beaker.org/ex/01KXE24SK50VY5RWX0WVBMRPEG) |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed3` | 8 | 16 | 3 | [01KXE24XX5DMZ573Z2GX378E95](https://beaker.org/ex/01KXE24XX5DMZ573Z2GX378E95) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed2` | 2 | 64 | 2 | [01KXE251AT69KM94WWAR77VAQ5](https://beaker.org/ex/01KXE251AT69KM94WWAR77VAQ5) |
| `2k_baseline_dapo_n2_k64_gradnorm1_seed3` | 2 | 64 | 3 | [01KXE254WXF2QEE7W0EG3VQHV4](https://beaker.org/ex/01KXE254WXF2QEE7W0EG3VQHV4) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed2` | 4 | 32 | 2 | [01KXE258GDHF9BSAJ621K68PQF](https://beaker.org/ex/01KXE258GDHF9BSAJ621K68PQF) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed3` | 4 | 32 | 3 | [01KXE25C47AH6RH09QSZZ55PG3](https://beaker.org/ex/01KXE25C47AH6RH09QSZZ55PG3) |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed4` | 4 | 32 | 4 | [01KXE25FQ2AGZ4R8J75C5BDP9C](https://beaker.org/ex/01KXE25FQ2AGZ4R8J75C5BDP9C) |

```bash
OC=true EXP=$NAME \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout $N --num_samples_per_prompt_rollout $K \
  --max_grad_norm 1.0 --seed $SEED
```

### Relaunch with `eval_step` metric fix (commit `c9d6c8453`)

Original 6 runs above (commit `263fdca36`) were canceled ~14% in and relaunched
fresh (not resumed — new `--checkpoint_state_dir`) after fixing `maybe_evaluate`
to log an explicit `eval_step` metric alongside eval results, so evals for a
given checkpoint step are always tagged with that step even if result
collection is deferred to a later training step. Same configs, still
`ai2/open-instruct-dev` workspace / `ai2/jupiter` cluster.

## n4_k32 gradnorm1 stall → rerun with async_steps 2

`2k_baseline_dapo_n4_k32_gradnorm1_seed1`
([01KX6ZC4NV9D7X7HJ1WR8WHS0P](https://beaker.org/ex/01KX6ZC4NV9D7X7HJ1WR8WHS0P))
entered the all-zero-reward filtering spin at step ~547 and crawled (13
steps/9h). Log analysis + wandb: completion lengths drifted up to the 8192
cap all run; `val/rho_weight` declined 1.0 → 0.9 over steps ~535–546 (growing
async off-policyness as long generations slowed, negative advantage dominating)
then snapped back to 1.0 post-collapse (steps became rare ⇒ on-policy again) —
so trainer and vLLM agree post-collapse, ruling out weight-sync/OLMo-core
optimizer corruption. Mechanism: length distribution crossed the truncation
cliff, ~60% of completions unfinished ⇒ nearly all groups all-zero ⇒ active
sampling filters everything. Canceled and rerun from scratch with
`--async_steps 2` (was 4) to bound the off-policy feedback loop; same image
(`michaeln/open-instruct-integration-test-ngu` @ `c9d6c8453`), workspace
`ai2/open-instruct-dev`.

| Name | n | k | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n4_k32_gradnorm1_async2_seed1` | 4 | 32 | 2 | 1 | [01KX9EKWT8Z7A0XJ7V70FC3D26](https://beaker.org/ex/01KX9EKWT8Z7A0XJ7V70FC3D26) |

### Launch command

```bash
OC=true EXP=2k_baseline_dapo_n4_k32_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 \
  --max_grad_norm 1.0 --seed 1 --async_steps 2
```

## NGU relaunch with rank-0 state-saving fix + async_steps 2

Codex fix (commit `7c8919d2f`): `DataPreparationActorCheckpointCallback` now
saves/restores the DataPreparationActor state (which holds the never_give_up
bookkeeping) from global rank 0 only. Previously all 4 FSDP ranks snapshotted
the live actor independently and all 4 restored on resume, so the last
`set_state` won with a potentially inconsistent snapshot.

Status of the gradnorm1 NGU runs at relaunch time: `ngu05` exited 1 at 97%,
`ngu075` exited 1 at 20%, `ngu0875` still running at 64% — killed. All three
relaunched from scratch on a rebuilt `michaeln/open-instruct-integration-test-ngu`
(commit `bdc538338`) with `--async_steps 2` added (see the n4_k32 stall
diagnosis above), workspace `ai2/open-instruct-dev`.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed1` | 8 | 16 | 0.5 | 2 | 1 | [01KX9FFKCW1NYPKAY16AXAWBM7](https://beaker.org/ex/01KX9FFKCW1NYPKAY16AXAWBM7) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed1` | 8 | 16 | 0.75 | 2 | 1 | [01KX9FG1G7G4BPHAHK2DANH727](https://beaker.org/ex/01KX9FG1G7G4BPHAHK2DANH727) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed1` | 8 | 16 | 0.875 | 2 | 1 | ~~[01KX9FGGQ5QGBTXGCKESYB8C91](https://beaker.org/ex/01KX9FGGQ5QGBTXGCKESYB8C91)~~ (killed 2026-07-12, relaunched) [01KXCH3WN17WXAWYW8FEGC4DY9](https://beaker.org/ex/01KXCH3WN17WXAWYW8FEGC4DY9) |

2026-07-12: the `ngu0875` run was still running but got killed and relaunched
from scratch on the freshly rebuilt image (commit `de6eb6fa6`), same config
and launch command.

### Launch command (repeat per row; first row built the image via build_image_and_launch.sh)

```bash
OC=true EXP=2k_ngu${P}_dapo_n8_k16_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up $P --async_steps 2
```

## 2026-07-13: 2 more NGU seeds per p, async_steps=4 (default), on ai2/titan

Parallel to the [2 more baseline seeds at async4](#2026-07-13-2-more-seeds-n8_k16-n2_k64--3-more-seeds-n4_k32-all-async_steps4-default):
seed2 + seed3 for each of `p ∈ {0.5, 0.75, 0.875}` at the default
`async_steps=4` (not the async2 mitigation used for the seed1 gradnorm1 NGU
relaunches above). These post-date the rank-0 state-saving fix (`7c8919d2f`),
so — unlike the original pre-fix async4 NGU seed1 runs — they're valid async4
data points to compare against the async2 seed1 runs. Same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, no code
changes), workspace `ai2/open-instruct-dev`, launched on `ai2/titan` (via
`CLUSTER=ai2/titan`) instead of the usual `ai2/jupiter`.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed2` | 8 | 16 | 0.5 | 4 | 2 | ~~[01KXE29J2GEQM5KPSKZRS528YP](https://beaker.org/ex/01KXE29J2GEQM5KPSKZRS528YP)~~ (stuck pending, moved) |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed3` | 8 | 16 | 0.5 | 4 | 3 | ~~[01KXE29NSETZEWKBXSW1Z8SXK1](https://beaker.org/ex/01KXE29NSETZEWKBXSW1Z8SXK1)~~ (stuck pending, moved) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed2` | 8 | 16 | 0.75 | 4 | 2 | ~~[01KXE29SA8CKYXMZAAC3MS89AB](https://beaker.org/ex/01KXE29SA8CKYXMZAAC3MS89AB)~~ (stuck pending, moved) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed3` | 8 | 16 | 0.75 | 4 | 3 | ~~[01KXE29WNXSPRP7BQCKJF4RYC4](https://beaker.org/ex/01KXE29WNXSPRP7BQCKJF4RYC4)~~ (stuck pending, moved) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed2` | 8 | 16 | 0.875 | 4 | 2 | ~~[01KXE2A09GXY5SSDR8X8NG5AVP](https://beaker.org/ex/01KXE2A09GXY5SSDR8X8NG5AVP)~~ (stuck pending, moved) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed3` | 8 | 16 | 0.875 | 4 | 3 | ~~[01KXE2A46X7SDXQ9BSAK5TWRWK](https://beaker.org/ex/01KXE2A46X7SDXQ9BSAK5TWRWK)~~ (stuck pending, moved) |

### Launch command (repeat per row)

```bash
OC=true EXP=2k_ngu${P}_dapo_n8_k16_gradnorm1_seed${SEED} \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/titan \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed $SEED --never_give_up $P
```

### 2026-07-13: moved to workspace `ai2/oe-adapt-code`, priority `high` (allocation didn't fit)

All 6 jobs above sat stuck in `pending` on `ai2/open-instruct-dev` (didn't fit
that workspace's allocation on `ai2/titan`). Stopped all 6 and relaunched
identically except `WORKSPACE=ai2/oe-adapt-code PRIORITY=high` (verified
first: `ai2/oe-adapt-code` is a real, writable-by-`michaeln` workspace, same
`ai2/oe-other` budget, whose max workload priority is exactly `high` — vs.
`ai2/open-instruct-dev`'s max of `urgent`). Same image, same `ai2/titan`
cluster. All 6 confirmed `starting` (not stuck `pending`) shortly after
launch.

| Name | Seed | Beaker |
| --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed2` | 2 | [01KXE31BCDT881YTHDVF2SZX2V](https://beaker.org/ex/01KXE31BCDT881YTHDVF2SZX2V) |
| `2k_ngu05_dapo_n8_k16_gradnorm1_seed3` | 3 | [01KXE31F1H6ERWBAR4SXRRW7KE](https://beaker.org/ex/01KXE31F1H6ERWBAR4SXRRW7KE) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed2` | 2 | [01KXE31JGTDMCB9BPFSCAH55ZH](https://beaker.org/ex/01KXE31JGTDMCB9BPFSCAH55ZH) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed3` | 3 | [01KXE31PB9B5R50MHY3PBYBVRK](https://beaker.org/ex/01KXE31PB9B5R50MHY3PBYBVRK) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed2` | 2 | [01KXE31SZJP0QNV1MR9V9K001V](https://beaker.org/ex/01KXE31SZJP0QNV1MR9V9K001V) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed3` | 3 | [01KXE31XDYEXFNNBNS0ETRXCH3](https://beaker.org/ex/01KXE31XDYEXFNNBNS0ETRXCH3) |

```bash
OC=true EXP=2k_ngu${P}_dapo_n8_k16_gradnorm1_seed${SEED} \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/oe-adapt-code CLUSTER=ai2/titan PRIORITY=high \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed $SEED --never_give_up $P
```

## Holmes (B300) cluster test: NGU 0.75 gradnorm1 async2

Test of the `ai2/holmes` cluster (B300 SXM6 nodes, Blackwell Ultra / sm_103)
with the existing CUDA 12.8 image `michaeln/open-instruct-integration-test-ngu`.
Same config as `2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed1`. Compatibility
notes: torch 2.10+cu128, flash-attn 2, and the vLLM 0.19.1 kernels all ship
plain `sm_100` SASS, which is binary-compatible with sm_103; FA3 ships only
sm_80/sm_90a kernels (no Blackwell) but is not selected on Blackwell —
`detect_attn_implementation` picks `flash_4` (JIT CuTe DSL) on compute
capability 10.x, so attention backends were left at their defaults. Residual
risk is any `sm_100a`-only kernel path (won't load on sm_103, "no kernel image
available"). Added `ai2/holmes` to `WEKA_CLUSTERS` in
`open_instruct/launch_utils.py` so mason mounts weka and sets the usual
checkpoint/output dirs.

First attempt ([01KX9H0REJ739F314KTSDQ21CY](https://beaker.org/ex/01KX9H0REJ739F314KTSDQ21CY))
confirmed the CUDA analysis — FA4 auto-selected, all kernels loaded — but died
in our own code: `get_device_name` raised on the unrecognized device string
`NVIDIA B300 SXM6 AC`. Added a `b300` entry to `GPU_SPECS` in
`open_instruct/utils.py` (288 GB HBM3e, 8 TB/s, dense BF16 ~2.25 PFLOPS same
as B200, commit `843258932`) and relaunched on a rebuilt image.

Second attempt ([01KX9HHB3KA075PKQ4DM886RXT](https://beaker.org/ex/01KX9HHB3KA075PKQ4DM886RXT))
got through FSDP setup but crashed in weight init: `tensor.erfinv_()` →
`nvrtc: error: invalid value for --gpu-architecture` — torch's jiterator ops
are runtime-compiled via NVRTC, and the NVRTC 12.8 pinned by torch cu128
can't target compute_103. Fixed by overriding `nvidia-cuda-nvrtc-cu12` to
12.9.86 (same `libnvrtc.so.12` soname; linux/x86_64 only) in pyproject
(commit `0b69301e7`). Triton / `torch.compile` needs no fix — it selects its
bundled 12.9 `ptxas-blackwell` for arch >= 100.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_holmes_seed1` | 8 | 16 | 0.75 | 2 | 1 | ~~[01KX9H0REJ739F314KTSDQ21CY](https://beaker.org/ex/01KX9H0REJ739F314KTSDQ21CY)~~ (GPU_SPECS crash) ~~[01KX9HHB3KA075PKQ4DM886RXT](https://beaker.org/ex/01KX9HHB3KA075PKQ4DM886RXT)~~ (NVRTC crash) TBD |

### Launch command

```bash
OC=true EXP=2k_ngu075_dapo_n8_k16_gradnorm1_async2_holmes_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/holmes \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2
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
| `ngu_quartiles_2gpu` (OC=false re-check) | Same smoke test, run directly through `grpo_fast.py --deepspeed_stage 2` to confirm NGU + per-quartile logging work on the DeepSpeed backend (investigating whether we can move NGU sweeps off `grpo.py`/OLMo-core, which has been unstable) | [01KXCV6XWK0KP20KFTB9769DEN](https://beaker.org/ex/01KXCV6XWK0KP20KFTB9769DEN) |

## OC=false (grpo_fast.py) NGU parity check

Audited whether `grpo_fast.py` needs any NGU/dataset-logging features ported
from `grpo.py`. Finding: `--never_give_up` (and all its
`maintain_pending_ngu_*` knobs), the rho-correction/TV-divergence masking, and
the dataset-specific eval logging (per-dataset `eval/pass_at_1/<label>`,
`eval/prompt_solve_rate_by_index_table`, per-quartile batch metrics) all live
in shared modules (`grpo_utils.py`, `data_loader.py`, `data_loader_utils.py`)
that both `grpo.py` and `grpo_fast.py` already consume identically via the
same `GRPOExperimentConfig`/`StreamingDataLoaderConfig` CLI dataclasses. No
code changes were needed. The only `grpo.py`-exclusive feature is
`--eval_only` (standalone eval round, no learner GPUs), which is unrelated to
training runs. Confirmed via the 2-GPU smoke test above (OC=false re-check)
before considering a full NGU sweep on the DeepSpeed backend.

## NGU 0.875, 12k response length (OC=true)

Testing a longer response length (12288 vs the usual 8192) with the leading
NGU `p=0.875` config, to see whether giving the model more room to finish
long completions changes the picture (recall the `N=4,K=32` gradnorm=1.0
collapse was triggered by completions drifting into the 8192 truncation
cap). `pack_length` scaled accordingly (14336 = 12288 + 2048 prompt).
Otherwise same config as the gradnorm=1.0 NGU sweep (seed 1, `async_steps 2`,
`max_grad_norm 1.0`, `total_episodes 256000`).

| Name | n | k | never_give_up | response_length | pack_length | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed1_resp12k` | 8 | 16 | 0.875 | 12288 | 14336 | 2 | 1 | [01KXCVGP6KGT8YEHZRGD8Q3G5C](https://beaker.org/ex/01KXCVGP6KGT8YEHZRGD8Q3G5C) |

## NGU gradnorm=1.0 p-sweep, OC=false (grpo_fast.py)

Replicates the [gradnorm=1.0 NGU sweep](#ngu-relaunch-with-rank-0-state-saving-fix--async_steps-2)
(same n8k16, seed 1, `async_steps 2`, `max_grad_norm 1.0`, `total_episodes
256000`) on the DeepSpeed backend instead of OLMo-core/FSDP, following the
[OC=false parity check](#ocfalse-grpo_fastpy-ngu-parity-check) above — `grpo.py`
has been unstable (FSDP rank-0 state-saving bug, B300 NVRTC issues), so
testing whether `grpo_fast.py` gives a cleaner comparison. Same image
(`michaeln/open-instruct-integration-test-ngu`, no code changes needed).

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed1` (OC=false) | 8 | 16 | 0.5 | 2 | 1 | [01KXCW3AB305HPFY1CH9ESM63N](https://beaker.org/ex/01KXCW3AB305HPFY1CH9ESM63N) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed1` (OC=false) | 8 | 16 | 0.75 | 2 | 1 | [01KXCW3ER1WHF6F13W736D22MH](https://beaker.org/ex/01KXCW3ER1WHF6F13W736D22MH) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed1` (OC=false) | 8 | 16 | 0.875 | 2 | 1 | [01KXCW3K087PWSNP7M57A8CHKZ](https://beaker.org/ex/01KXCW3K087PWSNP7M57A8CHKZ) |

### Launch command (repeat per row)

```bash
OC=false EXP=2k_ngu${P}_dapo_n8_k16_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up $P --async_steps 2
```

### Launch command

```bash
OC=true EXP=2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed1_resp12k \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.875 --async_steps 2 \
  --response_length 12288 --pack_length 14336
```

## 2026-07-14: NGU 0.5 seed 3 written off, new seed 4 (async2, jupiter, urgent) + holmes retry

`2k_ngu05_dapo_n8_k16_gradnorm1_seed3` ([01KXE31F1H6ERWBAR4SXRRW7KE](https://beaker.org/ex/01KXE31F1H6ERWBAR4SXRRW7KE),
wandb `8hsqittg`) came out bad — left running (79% done) but not trusted, not
used for comparison. Replaced with a fresh seed 4 at `--async_steps 2` (the
gradnorm1 NGU mitigation), launched on `ai2/jupiter` / `ai2/open-instruct-dev`
/ `urgent` priority — the combo that's been running cleanly for the other
NGU gradnorm1 seed1 relaunches, as opposed to the `ai2/titan` / `oe-adapt-code`
/ `high`-priority combo that's been getting preempted (see seed2 0.875 entry
below).

Also fired the same config at `ai2/holmes` (B300) again, out of curiosity —
the two infra fixes from the [holmes attempt](#holmes-b300-cluster-test-ngu-075-gradnorm1-async2)
(weka mount registration `843258932`, NVRTC 12.9 override `0b69301e7`) are
now both in this image/commit (`de6eb6fa6`), so this checks whether they're
sufficient for a full run rather than just clearing the two known crash points.

| Name | n | k | never_give_up | async_steps | Seed | Cluster | Workspace | Priority | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed4` | 8 | 16 | 0.5 | 2 | 4 | ai2/jupiter | ai2/open-instruct-dev | urgent | [01KXGQY2PVSM113ZWRRS3X4SKM](https://beaker.org/ex/01KXGQY2PVSM113ZWRRS3X4SKM) |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_holmes_seed4` | 8 | 16 | 0.5 | 2 | 4 | ai2/holmes | ai2/open-instruct-dev | urgent | [01KXGQYC430GA8V8TNF05EFMAB](https://beaker.org/ex/01KXGQYC430GA8V8TNF05EFMAB) |

### Launch commands

```bash
OC=true EXP=2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed4 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter PRIORITY=urgent \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 4 --never_give_up 0.5 --async_steps 2

OC=true EXP=2k_ngu05_dapo_n8_k16_gradnorm1_async2_holmes_seed4 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/holmes PRIORITY=urgent \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 4 --never_give_up 0.5 --async_steps 2
```

## 2026-07-14: NGU 0.875 seed 2 continued on same wandb run, moved to olmo-instruct/urgent

`2k_ngu0875_dapo_n8_k16_gradnorm1_seed2` ([01KXE31SZJP0QNV1MR9V9K001V](https://beaker.org/ex/01KXE31SZJP0QNV1MR9V9K001V),
wandb `l4ynoh5b`) looked promising (60.9% through, step 1218/2000) but was
stuck: repeatedly preempted on `ai2/titan`/`ai2/oe-adapt-code` at `high`
priority by other `urgent` jobs, and its 4th restart sat in `pending` without
scheduling at all. Stopped that experiment and relaunched a continuation of
the *same* checkpoint and the *same wandb run* (rather than starting a fresh
wandb id) on `ai2/olmo-instruct` / `urgent` priority (default cluster
`ai2/jupiter`).

Launched via a direct `mason.py` call (not the `qwen3_4b_deepscaler_math.sh`
wrapper, which can't pass through `--env`/`--non_resumable`) reusing the
exact training-script arguments from the stopped job, with `--non_resumable`
(so mason.py doesn't auto-generate its own fresh `WANDB_RUN_ID` and clobber
ours) and `--env WANDB_RUN_ID=l4ynoh5b --env WANDB_RESUME=allow` to resume
logging into the same wandb run. `--checkpoint_state_dir` was kept pointed at
the original run's checkpoint path (already under `/weka/`, so mason.py's
auto-checkpoint-dir override leaves it alone).

| Name | Seed | Old Beaker (stopped) | Workspace | Priority | New Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_seed2` | 2 | [01KXE31SZJP0QNV1MR9V9K001V](https://beaker.org/ex/01KXE31SZJP0QNV1MR9V9K001V) | ai2/olmo-instruct | urgent | [01KXGQZ0F0XV6C78RK8ACFNMYC](https://beaker.org/ex/01KXGQZ0F0XV6C78RK8ACFNMYC) |

Verified via `beaker experiment get` that the new job's env has
`WANDB_RUN_ID=l4ynoh5b`, `WANDB_RESUME=allow`, workspace `ai2/olmo-instruct`,
priority `urgent`, cluster `ai2/jupiter`.

### Launch command

```bash
uv run mason.py \
  --task_name qwen3_4b_base_deepscaler_oc_2k_ngu0875_dapo_n8_k16_gradnorm1_seed2 \
  --description "continued, same wandb l4ynoh5b" \
  --cluster ai2/jupiter --workspace ai2/olmo-instruct --priority urgent \
  --pure_docker_mode --no_auto_dataset_cache \
  --image michaeln/open-instruct-integration-test-ngu \
  --preemptible --num_nodes 1 --non_resumable \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env GIT_COMMIT=de6eb6fa6 \
  --env WANDB_RUN_ID=l4ynoh5b --env WANDB_RESUME=allow \
  --gpus 8 \
  -- source configs/beaker_configs/ray_node_setup.sh \
  && uv run open_instruct/grpo.py --run_name qwen3_4b_base_deepscaler_oc_2k_ngu0875_dapo_n8_k16_gradnorm1_seed2_20260713_115533 \
  --exp_name qwen3_4b_base_deepscaler_oc_2k_ngu0875_dapo_n8_k16_gradnorm1_seed2 \
  [... same args as original job ...] \
  --seed 2 --never_give_up 0.875 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1783958136_242059 \
  --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/
```

## 2026-07-14: baseline n8_k16 seed4, NGU 0.75/0.875 async2 seed2, and first n16_k8 gradnorm1 NGU (p=0.75) seed1

All on `ai2/jupiter` (script default), `ai2/open-instruct-dev` workspace, `urgent`
priority (script default), same image (`michaeln/open-instruct-integration-test-ngu`
@ `de6eb6fa6`, no code changes — launched directly via `bash`, no rebuild). Extends
the `n8_k16` gradnorm1 baseline to a 4th seed, adds a 2nd async2 seed for NGU
`p=0.75` and `p=0.875` (previously only seed1 existed at async2 for those two —
seed2 existed for both but at the default async_steps=4 on `ai2/titan`), and opens
up a new config: `n16_k8` had never been run at `max_grad_norm=1.0` before (only
the original grad_norm=5.0 sweep at p=0.5/0.9), so this is its first gradnorm1 NGU
run, seed1, `p=0.75`, `async_steps 2`.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed4` | 8 | 16 | — | 4 (default) | 4 | [01KXH36V5W69MQSGMKZ3WD7M6V](https://beaker.org/ex/01KXH36V5W69MQSGMKZ3WD7M6V) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed2` | 8 | 16 | 0.75 | 2 | 2 | [01KXH376J6R8Q7H3R3XP2XHT5D](https://beaker.org/ex/01KXH376J6R8Q7H3R3XP2XHT5D) |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed2` | 8 | 16 | 0.875 | 2 | 2 | [01KXH37D8AYKCGBMHFPJ9M62F0](https://beaker.org/ex/01KXH37D8AYKCGBMHFPJ9M62F0) |
| `2k_ngu075_dapo_n16_k8_gradnorm1_async2_seed1` | 16 | 8 | 0.75 | 2 | 1 | [01KXH38100EW4CEMTPA9FM6KDA](https://beaker.org/ex/01KXH38100EW4CEMTPA9FM6KDA) |

### Launch commands

```bash
OC=true EXP=2k_baseline_dapo_n8_k16_gradnorm1_seed4 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 4

OC=true EXP=2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed2 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 2 --never_give_up 0.75 --async_steps 2

OC=true EXP=2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed2 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 2 --never_give_up 0.875 --async_steps 2

OC=true EXP=2k_ngu075_dapo_n16_k8_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2
```

## 2026-07-15: NGU 0.875 async2 seed3, on ai2/titan

Third async2 seed for `p=0.875` (n8_k16, gradnorm1) — seed1 and seed2 already
exist at this config (see the [rank-0-fix async2 launch](#ngu-relaunch-with-rank-0-state-saving-fix--async_steps-2)
and [2026-07-14 seed2 batch](#2026-07-14-baseline-n8_k16-seed4-ngu-0750875-async2-seed2-and-first-n16_k8-gradnorm1-ngu-p075-seed1)).
This one explicitly targets `ai2/titan` (via `CLUSTER=ai2/titan`) instead of
the script's `ai2/jupiter` default, on workspace `ai2/open-instruct-dev`. Note:
the last time NGU jobs were launched on `ai2/titan`/`ai2/open-instruct-dev`
(see the [2026-07-13 titan batch](#2026-07-13-2-more-ngu-seeds-per-p-async_steps4-default-on-ai2titan))
they sat stuck `pending` and had to move to `ai2/oe-adapt-code`/`high`
priority — this one confirmed `starting` (not stuck) shortly after launch, so
that allocation issue doesn't appear to be recurring here. Same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, no code
changes), launched directly via `bash` (no rebuild needed).

| Name | n | k | never_give_up | async_steps | Seed | Cluster | Workspace | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed3` | 8 | 16 | 0.875 | 2 | 3 | ai2/titan | ai2/open-instruct-dev | [01KXJ2EJYAC6X2HNYXN9RJH7BW](https://beaker.org/ex/01KXJ2EJYAC6X2HNYXN9RJH7BW) |

### Launch command

```bash
OC=true EXP=2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed3 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/titan \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 3 --never_give_up 0.875 --async_steps 2
```

## 2026-07-14: n16_k8 gradnorm1 NGU p=0.825 seed1

Second `n16_k8` gradnorm1 NGU data point (after the `p=0.75` seed1 above), at
`p=0.825` — bracketing between `0.75` and `0.875`. Same image/workspace/cluster/
priority as the rest of the 2026-07-14 batch.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu0825_dapo_n16_k8_gradnorm1_async2_seed1` | 16 | 8 | 0.825 | 2 | 1 | [01KXH5W45NEPA1REXZPNEDXP2D](https://beaker.org/ex/01KXH5W45NEPA1REXZPNEDXP2D) |

### Launch command

```bash
OC=true EXP=2k_ngu0825_dapo_n16_k8_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.825 --async_steps 2
```

## 2026-07-16: n4_k32 baseline seed5, and first NGU 0.75 KL-penalty (beta=0.01) run

Two new jobs, both `ai2/jupiter` (script default), `ai2/open-instruct-dev`
workspace, `urgent` priority (script default), same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches current
`HEAD` — launched directly via `bash`, no rebuild).

- `n4_k32` gradnorm1 baseline gets a 5th seed (seeds 1/relaunched-d9z062ob,
  2, 3, 4 already exist).
- First NGU `p=0.75` run with a nonzero KL penalty (`--beta 0.01`, all prior
  NGU/baseline runs in this sweep use `--beta 0.0`). Note: the codebase
  rejects `beta != 0.0` unless `--load_ref_policy True` is also set (raises
  `ValueError` otherwise, see `grpo_utils.py`), so this run adds
  `--load_ref_policy True` on top of the usual `async_steps 2` NGU 0.75
  recipe. New sub-config, so it starts its own seed count at 1 rather than
  continuing the (beta=0.0) `p=0.75` async2 seed numbering.

| Name | n | k | never_give_up | async_steps | beta | load_ref_policy | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n4_k32_gradnorm1_seed5` | 4 | 32 | — | 4 (default) | 0.0 | False (default) | 5 | [01KXMVCH50K2898B0VDBJECY2E](https://beaker.org/ex/01KXMVCH50K2898B0VDBJECY2E) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_kl001_seed1` | 8 | 16 | 0.75 | 2 | 0.01 | True | 1 | [01KXMVD02C3ANQQQE9BVGPCA96](https://beaker.org/ex/01KXMVD02C3ANQQQE9BVGPCA96) |

### Launch commands

```bash
OC=true EXP=2k_baseline_dapo_n4_k32_gradnorm1_seed5 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 \
  --max_grad_norm 1.0 --seed 5

OC=true EXP=2k_ngu075_dapo_n8_k16_gradnorm1_async2_kl001_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2 --beta 0.01 --load_ref_policy True
```

## 2026-07-16: n8_k16 baseline seed5, on ai2/titan

5th seed for the `n8_k16` gradnorm1 baseline (seeds 1-4 already exist).
Explicitly targets `ai2/titan` (via `CLUSTER=ai2/titan`) instead of the
script's `ai2/jupiter` default, workspace `ai2/open-instruct-dev`, `urgent`
priority (script default). Same image (`michaeln/open-instruct-integration-test-ngu`
@ `de6eb6fa6`, matches current `HEAD`), launched directly via `bash`.
Confirmed `starting` (not stuck `pending`) shortly after launch — the
`ai2/titan`/`ai2/open-instruct-dev` allocation issue from the
[2026-07-13 titan batch](#2026-07-13-2-more-ngu-seeds-per-p-async_steps4-default-on-ai2titan)
isn't recurring here (consistent with the [2026-07-15 NGU 0.875 seed3 titan launch](#2026-07-15-ngu-0875-async2-seed3-on-ai2titan)).

| Name | n | k | async_steps | Seed | Cluster | Workspace | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_baseline_dapo_n8_k16_gradnorm1_seed5` | 8 | 16 | 4 (default) | 5 | ai2/titan | ai2/open-instruct-dev | [01KXMVK4YTHKJYT0E08DZDJFSE](https://beaker.org/ex/01KXMVK4YTHKJYT0E08DZDJFSE) |

### Launch command

```bash
OC=true EXP=2k_baseline_dapo_n8_k16_gradnorm1_seed5 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/titan \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 5
```

## 2026-07-16: NGU 0.5 async2 seed5

5th seed for `p=0.5` (seed3 was written off as bad, seed4 was the async2
jupiter/open-instruct-dev/urgent relaunch — see
[2026-07-14 NGU 0.5 seed 3 written off, new seed 4](#2026-07-14-ngu-05-seed-3-written-off-new-seed-4-async2-jupiter-urgent--holmes-retry)).
Same `--async_steps 2` gradnorm1 mitigation, `ai2/jupiter` (script default) /
`ai2/open-instruct-dev` / `urgent` (script default) — the combo that's been
running cleanly for these relaunches. Same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches current
`HEAD`), launched directly via `bash`.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed5` | 8 | 16 | 0.5 | 2 | 5 | [01KXMWSYTD0HCE83KA7B801ENX](https://beaker.org/ex/01KXMWSYTD0HCE83KA7B801ENX) |

### Launch command

```bash
OC=true EXP=2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed5 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 5 --never_give_up 0.5 --async_steps 2
```

## 2026-07-16: new initial-model difficulty eval, n=128 samples/prompt (replaces w47m67sf)

`notebooks/deepscaler_ngu_plots.ipynb`'s difficulty buckets (hard/medium/easy)
are derived from a single solve-rate eval of `Qwen3-4B-Base` on the
AIME+BRUMO quartile-sorted eval sets (`DIFFICULTY_RUN_ID = w47m67sf`,
64 samples/prompt). Relaunching at 128 samples/prompt for a less noisy
solve-rate estimate (and therefore a less noisy hard/medium/easy split) —
same base model, same two eval datasets, same ordering (AIME then BRUMO,
matching `w47m67sf`'s `dataset_mixer_eval_list` order so the notebook's
existing remap logic doesn't need to change).

Note: unlike `w47m67sf` (launched months ago on `grpo_fast.py`, which no
longer implements `--eval_only` — it's now only handled in
`open_instruct/grpo.py`, the OLMo-core script; see `grpo.py`'s
`run_eval_only`), this relaunch uses `OC=true` for that reason. `vllm` serves
`model_name_or_path` directly in eval-only mode (no OLMo-core model is
built), so pointing `--model_name_or_path` straight at the `Qwen/Qwen3-4B-Base`
HF hub name works without a training run. No `--eval_only_set_checkpoint` is
passed (matches `w47m67sf`, which also left it unset) since this evaluates
the base checkpoint, not a step from a training run.

| Name | eval samples/prompt | Beaker |
| --- | --- | --- |
| `dapo_evalonly_n128` | 128 | ~~[01KXMY55HA8T2PF7346SJ1QNT3](https://beaker.org/ex/01KXMY55HA8T2PF7346SJ1QNT3)~~ (stuck `pending` >1h on `ai2/olmo-instruct`, canceled) ~~[01KXN25GDJ8XRRH4Q09XVZQ57K](https://beaker.org/ex/01KXN25GDJ8XRRH4Q09XVZQ57K)~~ (scheduled fast but hit a cordoned/unhealthy node, "unrecoverable SXid error", auto-canceled before any code ran) ~~[01KXN28X40R2AYJTGS9PXWCCTB](https://beaker.org/ex/01KXN28X40R2AYJTGS9PXWCCTB)~~ (crashed on startup, async_steps assertion) ~~[01KXN2GAGXG0S7V5T43JBEJ9HK](https://beaker.org/ex/01KXN2GAGXG0S7V5T43JBEJ9HK)~~ (crashed on startup, unknown `--eval_temperature` flag) [01KXN2MX83YE6PKEMXYV868WJR](https://beaker.org/ex/01KXN2MX83YE6PKEMXYV868WJR) (finished cleanly, ~9 min runtime; wandb `79ol8lss`) |

**Result:** `notebooks/deepscaler_ngu_plots.ipynb`'s `DIFFICULTY_RUN_ID` set to
`79ol8lss`, `DIFFICULTY_NUM_SAMPLES` to 128. New hard/medium/easy counts
(AIME: 15/7/8, BRUMO: 13/8/9, Combined: 28/15/17) — see `research.md` for
comparison against the old 64-sample split.

The first launch (workspace `ai2/olmo-instruct`, the script default) sat
`pending` for over an hour on `ai2/jupiter` while everything else launched
today on `ai2/open-instruct-dev` (same cluster, same priority) started within
minutes — same workspace-allocation issue as prior `ai2/titan` cases (see
["moved to workspace `ai2/oe-adapt-code`"](#2026-07-13-moved-to-workspace-ai2oe-adapt-code-priority-high-allocation-didnt-fit)).
Canceled and relaunched with `WORKSPACE=ai2/open-instruct-dev`; that attempt
scheduled in seconds but landed on a hardware-faulty node and was
auto-canceled by Beaker before any of our code ran (unrelated to the eval
config). Relaunched a third time, identically — this one scheduled onto a
healthy node but crashed immediately on argument parsing: `--async_steps 1`
(matching `w47m67sf`'s original config, presumably from before this check
existed) now fails a `data_loader.py` assertion that `--active_sampling`
(on by default in `qwen3_4b_deepscaler_math.sh`) requires `async_steps > 1`.
Fixed by bumping to `--async_steps 2` on the 4th attempt — `eval_only` mode
(`run_eval_only` in `grpo.py`) runs a single evaluation round regardless of
`async_steps`, so this shouldn't affect the eval results, only satisfies the
parse-time check. That attempt crashed too, immediately, on a *different*
parse error: `--eval_temperature` no longer exists as a field at all (removed
from `grpo_utils.py`'s `GRPOExperimentConfig` since `w47m67sf` was launched
months ago) — `HfArgumentParser` rejects unknown flags outright. Per
`grpo_fast.py`'s `create_generation_configs`, eval generation now just reuses
the shared `--temperature` (only `top_p`/`n`/`max_tokens` get eval-specific
overrides via `eval_top_p`/`eval_pass_at_k`/`eval_response_length`). Fixed on
the 5th attempt by passing `--temperature 0.7` instead (harmless here since
`eval_only` never runs the training generation config).

### Launch command

```bash
OC=true EXP=dapo_evalonly_n128 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --eval_only --eval_pass_at_k 128 --temperature 0.7 --eval_top_p 0.9 --eval_response_length 8192 \
  --async_steps 2 \
  --dataset_mixer_eval_list mnoukhov/aime-2025-openinstruct-qwen3-4b-base-32samples-quartiles 1.0 mnoukhov/brumo-2025-openinstruct-qwen3-4b-base-32samples-quartiles 1.0 \
  --dataset_mixer_eval_list_splits train
```

## 2026-07-16: NGU 0.75 seed3 (gz2ux8w0) resumed from checkpoint after gloo comms crash

`2k_ngu075_dapo_n8_k16_gradnorm1_seed3` ([01KXE31PB9B5R50MHY3PBYBVRK](https://beaker.org/ex/01KXE31PB9B5R50MHY3PBYBVRK),
wandb [`gz2ux8w0`](https://wandb.ai/ai2-llm/open_instruct_internal/runs/gz2ux8w0),
`ai2/titan`/`ai2/oe-adapt-code`/`high`) ran cleanly for ~7.5h (step 0 → 948/2000,
`eval/pass_at_1` aime=0.205 brumo=0.290 at the last logged eval, step 802) then
died with `exitCode=1`: a Ray/gloo `RuntimeError: Connection closed by peer`
during an `all_reduce` inside `olmo_core`'s metric-reduction step — a transient
distributed-comms fault, not a code or config bug (no prior preemption pattern
for this job, unlike the [NGU 0.875 seed2 case](#2026-07-14-ngu-0875-seed-2-continued-on-same-wandb-run-moved-to-olmoinstructurgent)).

Resumed rather than relaunching from scratch: `checkpoint_state_freq=100` means
a checkpoint should exist at/near step 900. Confirmed the resume mechanism is
automatic — `grpo_olmo_core_actor.py` builds the OLMo-core `TrainerConfig` with
`save_folder=checkpoint_state_dir` and `load_strategy=LoadStrategy.if_available`,
and `grpo_utils.py`'s `__post_init__` calls `calibrate_checkpoint_state_dir()` on
startup to repair/point `latest` at the newest complete checkpoint — so passing
the same `--checkpoint_state_dir` is sufficient to resume training state.
Followed the [NGU 0.875 seed2 continuation precedent](#2026-07-14-ngu-0875-seed-2-continued-on-same-wandb-run-moved-to-olmoinstructurgent):
launched via a direct `mason.py` call (not the `qwen3_4b_deepscaler_math.sh`
wrapper) reusing the exact training-script args from the crashed job, with
`--non_resumable` + `--env WANDB_RUN_ID=gz2ux8w0 --env WANDB_RESUME=allow` to
keep logging into the same wandb run (so the notebook's per-run `best_step`
`scan_history()` lookup sees one continuous curve), same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches current
`HEAD` — no rebuild needed), same cluster/workspace/priority as the original
job since there was no preemption pattern here to avoid.

| Name | Seed | Old Beaker (crashed) | New Beaker |
| --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed3` | 3 | [01KXE31PB9B5R50MHY3PBYBVRK](https://beaker.org/ex/01KXE31PB9B5R50MHY3PBYBVRK) | [01KXNJN3V42VCH0ZJ8T0A36J79](https://beaker.org/ex/01KXNJN3V42VCH0ZJ8T0A36J79) |

### Launch command

```bash
uv run mason.py \
  --task_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed3 \
  --description "continued, same wandb gz2ux8w0" \
  --cluster ai2/titan --workspace ai2/oe-adapt-code --priority high \
  --pure_docker_mode --no_auto_dataset_cache \
  --image michaeln/open-instruct-integration-test-ngu \
  --preemptible --num_nodes 1 --non_resumable \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env GIT_COMMIT=de6eb6fa6 \
  --env WANDB_RUN_ID=gz2ux8w0 --env WANDB_RESUME=allow \
  --gpus 8 \
  -- source configs/beaker_configs/ray_node_setup.sh \
  && uv run open_instruct/grpo.py --run_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed3_20260713_115530 \
  --exp_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed3 \
  [... same training args as the original job ...] \
  --seed 3 --never_give_up 0.75 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1783958132_505819 \
  --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/
```

The first attempt ([01KXNJN3V42VCH0ZJ8T0A36J79](https://beaker.org/ex/01KXNJN3V42VCH0ZJ8T0A36J79),
`ai2/titan`/`ai2/oe-adapt-code`/`high`) sat `pending` with no `scheduled`
timestamp at all for ~55 minutes — same workspace-allocation stall pattern as
the [n=128 difficulty eval](#2026-07-16-new-initial-model-difficulty-eval-n128-samplesprompt-replaces-w47m67sf)
and the original [async4 NGU seeds on titan](#2026-07-13-2-more-ngu-seeds-per-p-async_steps4-default-on-ai2titan).
Stopped it and relaunched identically except `CLUSTER=ai2/jupiter
WORKSPACE=ai2/open-instruct-dev PRIORITY=urgent` (everything else launched
today on that combo scheduled within minutes).

| Name | Seed | Attempt 1 (stuck pending, moved) | Attempt 2 |
| --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed3` | 3 | ~~[01KXNJN3V42VCH0ZJ8T0A36J79](https://beaker.org/ex/01KXNJN3V42VCH0ZJ8T0A36J79)~~ | [01KXNNQC1EEQ2KZPTP0036A0JD](https://beaker.org/ex/01KXNNQC1EEQ2KZPTP0036A0JD) |

**Resume confirmed:** attempt 2 started cleanly and logs show
`training_step=901` in `accumulate_inference_batches` — resumed from the
step-900 checkpoint (the last one saved before the step-948 crash, matching
`checkpoint_state_freq=100`), not restarted from scratch. Still training as
of this writing; will update again once finalized/crashed.

## 2026-07-16: NGU 0.75 seed2 (wf6ttda7) resumed after repeated preemption

`2k_ngu075_dapo_n8_k16_gradnorm1_seed2` ([01KXE31JGTDMCB9BPFSCAH55ZH](https://beaker.org/ex/01KXE31JGTDMCB9BPFSCAH55ZH),
wandb [`wf6ttda7`](https://wandb.ai/ai2-llm/open_instruct_internal/runs/wf6ttda7),
`ai2/titan`/`ai2/oe-adapt-code`/`high`) — same original launch batch as the
`gz2ux8w0` case above. Got to 76% (step 1521/2000, aime=0.224/brumo=0.301 at
step 1503) but was repeatedly preempted by `urgent`-priority jobs at `high`
priority (7 restarts over ~13h, same pattern as the
[NGU 0.875 seed2 case](#2026-07-14-ngu-0875-seed-2-continued-on-same-wandb-run-moved-to-olmoinstructurgent));
the final restart never scheduled and was manually canceled. wandb state:
`crashed`.

Resumed directly on `ai2/jupiter`/`ai2/open-instruct-dev`/`urgent` (per user
request, skipping the titan/oe-adapt-code loop entirely this time) — same
`mason.py` pattern as the `gz2ux8w0` resume: reused the crashed job's exact
training args, `--checkpoint_state_dir` pointed at the original path, and
`--env WANDB_RUN_ID=wf6ttda7 --env WANDB_RESUME=allow` + `--non_resumable` to
keep the same wandb run.

| Name | Seed | Old Beaker (preempted repeatedly) | New Beaker |
| --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_gradnorm1_seed2` | 2 | [01KXE31JGTDMCB9BPFSCAH55ZH](https://beaker.org/ex/01KXE31JGTDMCB9BPFSCAH55ZH) | [01KXP317JMEGZSYGS66VE95NWG](https://beaker.org/ex/01KXP317JMEGZSYGS66VE95NWG) |

### Launch command

```bash
uv run mason.py \
  --task_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed2 \
  --description "continued, same wandb wf6ttda7, moved to jupiter/open-instruct-dev/urgent" \
  --cluster ai2/jupiter --workspace ai2/open-instruct-dev --priority urgent \
  --pure_docker_mode --no_auto_dataset_cache \
  --image michaeln/open-instruct-integration-test-ngu \
  --preemptible --num_nodes 1 --non_resumable \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env GIT_COMMIT=de6eb6fa6 \
  --env WANDB_RUN_ID=wf6ttda7 --env WANDB_RESUME=allow \
  --gpus 8 \
  -- source configs/beaker_configs/ray_node_setup.sh \
  && uv run open_instruct/grpo.py --run_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed2_20260713_115526 \
  --exp_name qwen3_4b_base_deepscaler_oc_2k_ngu075_dapo_n8_k16_gradnorm1_seed2 \
  [... same training args as the original job ...] \
  --seed 2 --never_give_up 0.75 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1783958128_223623 \
  --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/
```

**Resume confirmed:** started cleanly on `jupiter`/`open-instruct-dev` and
logs show `training_step=1525` in `accumulate_inference_batches` — resumed
right where the last preemption left off (step 1521), not restarted from
scratch. Still training as of this writing.

## 2026-07-16: refreshed best_step across all registered runs, swapped NGU 0.5 seed3 to zg0thiuz

Re-checked `best_step` (peak combined AIME+BRUMO pass@1) for all 18
`RUN_SPECS` entries in `notebooks/deepscaler_ngu_plots.ipynb` against current
wandb data. Only one changed materially: `NGU p=0.75` seed3 (`gz2ux8w0`,
resumed [above](#2026-07-16-ngu-075-seed3-gz2ux8w0-resumed-from-checkpoint-after-gloo-comms-crash))
now has eval data through step 1002 (vs. 802 before the resume), moving its
best_step from 800 to 1000. Everything else matched the recorded value
(within the round-to-nearest-100 convention) or is a deliberate override that
was preserved as-is: `N=4,K=32` seed2 (`617qdmx6`, pinned to 900 per the
easy-subset request) and `NGU p=0.875` seed1 (`pq8nul2d`, pinned to 1200 per
the hard-subset request).

Also swapped `NGU p=0.5` seed3 from `2sti5i22` to
[`zg0thiuz`](https://wandb.ai/ai2-llm/open_instruct_internal/runs/zg0thiuz)
(Beaker [01KXMWSYTD0HCE83KA7B801ENX](https://beaker.org/ex/01KXMWSYTD0HCE83KA7B801ENX),
the [NGU 0.5 async2 seed5 run launched earlier today](#2026-07-16-ngu-05-async2-seed5)
— underlying training `--seed 5`, kept in the "seed 3" display slot). Still
running, only 8 eval points through step 802 so far (best_step=800); will
need refreshing again as it progresses.

Re-executed the notebook (`jupyter nbconvert --execute --inplace`), 0 error
cells.

## 2026-07-17: NGU 0.75 seed4 (async2) + first KL-penalty attempt at beta=0.001

Two new jobs, both `ai2/jupiter` (explicit, matches script default),
`ai2/open-instruct-dev` workspace, `urgent` priority (script default), same
image (`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches
current `HEAD` — launched directly via `bash`, no rebuild).

- `p=0.75` gradnorm1 gets a 4th seed at the `--async_steps 2` NGU mitigation
  (seed1 `11dc2uid` and the two async4 titan/oe-adapt-code seeds `wf6ttda7`/
  `gz2ux8w0` already exist; a separate untracked async2 seed2
  (`2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed2`, Beaker `01KXH376J6R8Q7H3R3XP2XHT5D`)
  also exists from 2026-07-14 but was never wired into the notebook's
  `RUN_SPECS` — not touched here).
- Second NGU `p=0.75` KL-penalty attempt, this time `--beta 0.001` (an order
  of magnitude below the first attempt's `--beta 0.01`,
  [`2k_ngu075_dapo_n8_k16_gradnorm1_async2_kl001_seed1`](#2026-07-16-n4_k32-baseline-seed5-and-first-ngu-075-kl-penalty-beta001-run)).
  Same `--load_ref_policy True` requirement (the codebase rejects nonzero
  `--beta` otherwise). New sub-config, starts its own seed count at 1.

| Name | n | k | never_give_up | async_steps | beta | load_ref_policy | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed4` | 8 | 16 | 0.75 | 2 | 0.0 | False (default) | 4 | [01KXQ4TMNQJWGBTNAFY2T9WHPM](https://beaker.org/ex/01KXQ4TMNQJWGBTNAFY2T9WHPM) |
| `2k_ngu075_dapo_n8_k16_gradnorm1_async2_kl0001_seed1` | 8 | 16 | 0.75 | 2 | 0.001 | True | 1 | [01KXQ4TZ69TWE7V23EE023SHYE](https://beaker.org/ex/01KXQ4TZ69TWE7V23EE023SHYE) |

### Launch commands

```bash
OC=true EXP=2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed4 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 4 --never_give_up 0.75 --async_steps 2

OC=true EXP=2k_ngu075_dapo_n8_k16_gradnorm1_async2_kl0001_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2 --beta 0.001 --load_ref_policy True
```

Both confirmed `starting` (not stuck `pending`) via `beaker experiment get`
shortly after launch.

## 2026-07-17: NGU 0.75 seed2 (wf6ttda7) finished cleanly — crash-signature alert was teardown noise, plus a stale-cache bug found in the process

The background monitor watching the resumed `wf6ttda7` job
([01KXP317JMEGZSYGS66VE95NWG](https://beaker.org/ex/01KXP317JMEGZSYGS66VE95NWG),
see [seed2 resume above](#2026-07-16-ngu-075-seed2-wf6ttda7-resumed-after-repeated-preemption))
fired on a gloo `unbound_buffer` timeout signature in the tail logs. Checked
`beaker experiment get`: `exitCode: 0`, description
`"100.0% complete (step 2000/2000), finished in 9h 9m"` — the job actually
finished successfully; the gloo timeout is post-completion NCCL/gloo process
teardown noise, not a mid-training failure. wandb `wf6ttda7` confirms
`state=finished`, 21 eval rows through step 2000, best step still 1600
(0.2719 combined AIME+BRUMO) — unchanged, no new peak near the end.

While refreshing the notebook's `RUN_SPECS` comment for this run, hit a
caching bug in `download_run_history()` (`deepscaler_ngu_plots.ipynb`):
`.wandb_cache/wf6ttda7_history.json` had been written on 2026-07-14 while the
run was transiently in a non-`"running"` state (`crashed`, mid-preemption,
only 15 eval rows through step 1500) — the cache function only skips
re-fetching when `run.state != "running"`, so once the run later resumed and
then finished, it kept satisfying that check and silently served the stale
15-row snapshot forever, tripping the `best_step` row-count assertion
(`found 0` for `eval_step=1600`). Deleted the stale cache file and
re-executed; resolved. **This same failure mode can recur for any other run
that was ever cached mid-crash/mid-preemption before a later resume** (e.g.
`gz2ux8w0` had a similar crash-then-resume history) — worth either deleting
`.wandb_cache/*_history.json` for those run_ids preemptively or hardening
`download_run_history()` to only cache on a genuinely terminal state
(`finished`/`failed`, not `crashed`/`preempting`) rather than `!= "running"`.

## 2026-07-17: NGU 0.5 async2 seed6

6th seed for `p=0.5` (seed3 written off, seed4 is an async2 run launched
2026-07-14 that was never wired into the notebook's `RUN_SPECS` — analogous
to the untracked `p=0.75` async2 seed2 noted in the
[seed4/kl0001 entry](#2026-07-17-ngu-075-seed4-async2--first-kl-penalty-attempt-at-beta0001)
above — and seed5 is `zg0thiuz`). Same `--async_steps 2` gradnorm1
mitigation, `ai2/jupiter` (script default) / `ai2/open-instruct-dev` /
`urgent` (script default). Same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches current
`HEAD`), launched directly via `bash`, no rebuild.

| Name | n | k | never_give_up | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6` | 8 | 16 | 0.5 | 2 | 6 | [01KXQ6MN3686SHT4Q9PWETFSP2](https://beaker.org/ex/01KXQ6MN3686SHT4Q9PWETFSP2) |

### Launch command

```bash
OC=true EXP=2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 6 --never_give_up 0.5 --async_steps 2
```

Confirmed `starting` (not stuck `pending`) via `beaker experiment get`
shortly after launch.

## 2026-07-17: NGU 0.75 seed3 (gz2ux8w0) also finished cleanly — same false-alarm + stale-cache pattern as seed2

Background monitor on `gz2ux8w0`'s resumed job fired the same gloo
`unbound_buffer` timeout signature as [seed2 above](#2026-07-17-ngu-075-seed2-wf6ttda7-finished-cleanly--crash-signature-alert-was-teardown-noise-plus-a-stale-cache-bug-found-in-the-process).
Same resolution: `beaker experiment get` shows `exitCode: 0`, "100.0%
complete (step 2000/2000), finished in 18h 40m" — benign post-completion
teardown noise, not a real crash. wandb `gz2ux8w0` confirms `state=finished`,
21 eval rows through step 2000, best step still 1000 (0.2557 combined
AIME+BRUMO) even across the full run.

Note: the job that finished, `01KXNNQC1EEQ2KZPTP0036A0JD`, is *not* the
`New Beaker` id already on record from the
[2026-07-16 crash/resume entry](#2026-07-16-ngu-075-seed3-gz2ux8w0-resumed-from-checkpoint-after-gloo-comms-crash)
(`01KXNJN3V42VCH0ZJ8T0A36J79`) — there was at least one further resume of the
same wandb run between that entry and this one that never got logged here
(not captured in this conversation; presumably handled in an earlier session
via the same resume mechanism). Updated the notebook's `RUN_SPECS` `beaker`
field to the final `01KXNNQC1EEQ2KZPTP0036A0JD`.

Also hit the exact same stale-`.wandb_cache` bug flagged as a risk in the
seed2 entry: `.wandb_cache/gz2ux8w0_history.json` was frozen from 2026-07-14
(9 eval rows, only through step 900, predating even the first crash/resume).
Deleted it and re-executed the notebook; resolved, best_step confirmed
unchanged.

Two-for-two was enough evidence to fix it properly rather than keep
firefighting: changed `download_run_history()`'s caching condition from
`run.state != "running"` to `run.state == "finished"` — `"crashed"`,
`"failed"`, and `"preempting"` states in this project routinely get resumed
from checkpoint into the *same* wandb run id (that's the whole point of the
resume workflow used throughout this sweep), so caching on any non-`"running"`
state was silently freezing history the moment a run first crashed, even
though it could keep growing after a resume. Now only a genuinely terminal
`"finished"` state gets cached; anything else always refetches fresh, same as
`"running"` already did. Re-executed the full notebook, 0 error cells.

## 2026-07-17: NGU 0.5 seed3 (zg0thiuz) finished cleanly, best_step peak shifted

`zg0thiuz` (NGU p=0.5, seed 3, Beaker `01KXMWSYTD0HCE83KA7B801ENX`) finished
cleanly: `beaker experiment get` shows "100.0% complete (step 2000/2000),
finished in 1d 10h 7m", exit code 0. wandb state is `finished`,
`lastHistoryStep=2000`.

Unlike the two NGU 0.75 false-alarm cases, this one wasn't just a status
refresh: with the full run now in, its true peak moved. The registry had
`best_step=1000` (combined AIME+BRUMO pass_at_1 0.2490) from when the run was
still in progress, but the fuller history shows a later, higher peak at step
1700 (combined 0.2542: AIME 0.2229, BRUMO 0.2854), beating both step 1000 and
the intervening steps (1800-2000 dip well below, e.g. 0.1928 at step 1800).
Updated `RUN_SPECS`' `best_step` from 1000 to 1700 and the comment
accordingly.

No stale cache existed for this run yet (never previously cached under the
new `state == "finished"` condition), so this was a clean single fetch —
confirms the caching fix from the gz2ux8w0 entry is working as intended
going forward. Re-executed the full notebook, 0 error cells.

## 2026-07-17: NGU 0.75 seed3 swapped gz2ux8w0 -> cjr9kfxa (better on hard subset)

Checked whether `cjr9kfxa` (the KL beta=0.001 variant launched earlier today,
Beaker `01KXQ4TZ69TWE7V23EE023SHYE`, underlying `--seed 1`, `async_steps=2`)
beats the worst-performing registered NGU p=0.75 seed on the "hard"
difficulty bucket, as a candidate replacement.

Computed each registered p=0.75 seed's combined AIME+BRUMO hard-bucket
solve_rate at its own best_step:
- seed1 (11dc2uid, best_step=2000): hard=0.0167
- seed2 (wf6ttda7, best_step=1600): hard=0.0335
- seed3 (gz2ux8w0, best_step=1000): hard=0.0156 -- worst

Then swept `cjr9kfxa`'s available eval history (still running, only through
step 1100/2000) for its own hard-bucket peak: 0.0011-0.0435 across steps
100-1100, peaking at **step 700 (hard=0.0435)**. That beats all three
registered seeds, including gz2ux8w0 (the worst, 0.0156), so the swap
condition was met.

Replaced the seed3 slot: `gz2ux8w0` -> `cjr9kfxa`, `best_step` 1000 -> 700.
Since cjr9kfxa is still early (1100/2000 steps), this best_step is only its
peak-so-far and will need refreshing once the run progresses further --
flagged in the RUN_SPECS comment. Re-executed the full notebook, 0 error
cells.

## 2026-07-18: NGU 0.5 seed6 continued 2000->3000 steps

`2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6` finished cleanly at its original
2000-step target (Beaker
[01KXQ6MN3686SHT4Q9PWETFSP2](https://beaker.org/ex/01KXQ6MN3686SHT4Q9PWETFSP2),
wandb `8fpsebxl`, exit code 0). Extended it another 1000 steps (3000 total)
to see whether the NGU curve keeps climbing or has already plateaued/overfit
by step 2000.

The OLMo-core trainer (`grpo_olmo_core_actor.py`) uses `load_strategy =
LoadStrategy.if_available` against `checkpoint_state_dir` and a `max_duration`
computed from `num_training_steps` (itself `total_episodes // (n*k)`, shared
via `grpo_fast.setup_runtime_variables`, which `grpo.py` reuses) — so resuming
further than the original target is just: point `--checkpoint_state_dir` at
the same path, bump `--total_episodes` from 256000 to 384000 (3000 steps at
128 completions/step), and relaunch. No code changes needed. Kept the same
`--run_name`/`--exp_name`/`--seed 6` and reused the wandb run id (`8fpsebxl`)
via `--env WANDB_RUN_ID=8fpsebxl --env WANDB_RESUME=allow` with
`--non_resumable` (so mason.py doesn't auto-generate a fresh run id), same
pattern as the [NGU 0.875 seed2 continuation](#2026-07-14-ngu-0875-seed-2-continued-on-same-wandb-run-moved-to-olmo-instructurgent).
Launched via a direct `mason.py` call (the `qwen3_4b_deepscaler_math.sh`
wrapper can't pass `--env`/`--non_resumable` through). Same image
(`michaeln/open-instruct-integration-test-ngu` @ `de6eb6fa6`, matches current
`HEAD`), `ai2/jupiter` / `ai2/open-instruct-dev` / `urgent` (same combo the
original run used).

Note: the first launch attempt accidentally ran the `uv run
open_instruct/grpo.py ...` half of the command *locally* instead of inside
the Beaker job — the shell's unescaped `&&` split the command instead of
passing it through as one literal argument to mason.py's `--`. Caught
immediately (local run failed on `/weka` permission error, and the
Beaker job it did launch was only running the `source
ray_node_setup.sh` half). Stopped that stray job and relaunched with `\&\&`
(matching the wrapper script's own escaping) so the whole `source ... &&
uv run ...` string reaches mason.py as a single command.

| Name | Seed | Original Beaker (2000 steps, done) | New target | New Beaker |
| --- | --- | --- | --- | --- |
| `2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6` | 6 | [01KXQ6MN3686SHT4Q9PWETFSP2](https://beaker.org/ex/01KXQ6MN3686SHT4Q9PWETFSP2) | 3000 steps | [01KXVMKY6M1TT6EPN6TVWM5SCJ](https://beaker.org/ex/01KXVMKY6M1TT6EPN6TVWM5SCJ) |

Confirmed clean resume from the log: `[DataPreparationActor] Restored state:
training_step=2000, last_consumed_step=1999` — picked up exactly where the
original run left off rather than restarting from scratch.

### Launch command

```bash
uv run mason.py \
  --task_name qwen3_4b_base_deepscaler_oc_2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6 \
  --description "continued 2000->3000 steps, same wandb 8fpsebxl" \
  --cluster ai2/jupiter --workspace ai2/open-instruct-dev --priority urgent \
  --pure_docker_mode --no_auto_dataset_cache \
  --image michaeln/open-instruct-integration-test-ngu \
  --preemptible --num_nodes 1 --non_resumable \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env GIT_COMMIT=de6eb6fa6 \
  --env WANDB_RUN_ID=8fpsebxl --env WANDB_RESUME=allow \
  --gpus 8 \
  -- source configs/beaker_configs/ray_node_setup.sh \
&& uv run open_instruct/grpo.py --run_name qwen3_4b_base_deepscaler_oc_2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6_20260717_005123 \
  --exp_name qwen3_4b_base_deepscaler_oc_2k_ngu05_dapo_n8_k16_gradnorm1_async2_seed6 \
  [... same args as original job ...] \
  --seed 6 --never_give_up 0.5 --async_steps 2 \
  --total_episodes 384000 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1784263889_375104 \
  --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/
```

## 2026-07-18: NGU 0.75 seed3 swapped cjr9kfxa -> rg4fgf84 (dropped the KL variant)

Swapped the `notebooks/deepscaler_ngu_plots.ipynb` seed3 slot for `NGU
p=0.75` from `cjr9kfxa` (the `--beta 0.001` KL-penalty variant) to `rg4fgf84`
— the plain seed4 run (`2k_ngu075_dapo_n8_k16_gradnorm1_async2_seed4`, Beaker
[01KXQ4TMNQJWGBTNAFY2T9WHPM](https://beaker.org/ex/01KXQ4TMNQJWGBTNAFY2T9WHPM),
underlying `--seed 4`, `async_steps=2`, no KL) — to keep this slot a clean
seed replicate consistent with how every other setting's 3 seeds are picked,
rather than a KL variant that happened to win on the hard bucket. Finished
cleanly (99.7%/step 1993/2000 at last check). `best_step` set to 1500, its
own combined AIME+BRUMO `eval/pass_at_1` peak (0.2698).

Re-executed the full notebook after this edit (had initially only edited the
`RUN_SPECS` cell source without re-running, so the rendered plots/tables
still showed stale `cjr9kfxa` data — caught when asked to confirm). 0 error
cells after re-execution.

## 2026-07-19: NGU 0.75 seed3 swapped back rg4fgf84 -> cjr9kfxa (best on easy bucket)

Checked which of the three registered NGU p=0.75 seeds is best on the
"easy" difficulty bucket (combined AIME+BRUMO, each at its own best_step):
seed1 (`11dc2uid`) 0.7402, seed2 (`wf6ttda7`) 0.7422, seed3 (`rg4fgf84`)
**0.7832** — rg4fgf84 was the best. Replaced it with `cjr9kfxa` (the KL
beta=0.001 variant, Beaker
[01KXQ4TZ69TWE7V23EE023SHYE](https://beaker.org/ex/01KXQ4TZ69TWE7V23EE023SHYE),
underlying `--seed 1`) per direct request, now finished cleanly (2000/2000,
wandb state `finished`). `best_step` set to 800, its own combined
AIME+BRUMO `eval/pass_at_1` peak (0.2589).

Re-executed the full notebook immediately after this edit — 0 error cells.

## 2026-07-23: K ablation holding N*(1-p) and N*K fixed — n16_k8 p=0.875, n32_k4 p=0.9375

New idea: ablate `K` (`num_samples_per_prompt_rollout`) while holding
`N*K = 128` fixed (as all configs in this sweep already do) *and* also
holding `N*(1-p)` fixed at 2, where `p` is `--never_give_up`. The existing
`n8_k16 p=0.75` line is the first point on this curve
(`N*(1-p) = 8*0.25 = 2`); this adds the next two points:
`n16_k8 p=0.875` (`16*0.125 = 2`) and `n32_k4 p=0.9375` (`32*0.0625 = 2`).
`n32_k4` is a brand-new `N*K` config — never run before at any `p` or as a
baseline. Per user decision, launched only the two NGU runs (no matching
gradnorm1 baselines for `n16_k8`/`n32_k4` for now — `n16_k8` only has the
older grad_norm=5.0 baseline, `n32_k4` has none at all).

Both on `ai2/jupiter`, `ai2/open-instruct-dev` workspace, `urgent` priority
(script default), same image (`michaeln/open-instruct-integration-test-ngu`
@ `de6eb6fa6` — HEAD is `b1c2fde73` but the only diff since the image build
is `experiment.md`/`research.md`, so no rebuild needed), launched directly
via `bash`. Both `async_steps 2` (standard NGU mitigation), `max_grad_norm
1.0`, seed 1.

| Name | n | k | never_give_up | N*(1-p) | async_steps | Seed | Beaker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed1` | 16 | 8 | 0.875 | 2 | 2 | 1 | [01KY6SYEJVRAC71QWBRSV3AAM6](https://beaker.org/ex/01KY6SYEJVRAC71QWBRSV3AAM6) |
| `2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed1` | 32 | 4 | 0.9375 | 2 | 2 | 1 | [01KY6SYRR473W6C0CSWACHXAGA](https://beaker.org/ex/01KY6SYRR473W6C0CSWACHXAGA) |

### Launch commands

```bash
OC=true EXP=2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.875 --async_steps 2

OC=true EXP=2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 32 --num_samples_per_prompt_rollout 4 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.9375 --async_steps 2
```

## 2026-07-23: K ablation seeds 2 & 3 (n16_k8 p=0.875, n32_k4 p=0.9375)

Two more seeds each for the two new K-ablation points above, same image/
workspace/cluster/priority, no rebuild.

| Name | n | k | never_give_up | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed2` | 16 | 8 | 0.875 | 2 | [01KY848DSJBK6WPEQVZY7VNCK5](https://beaker.org/ex/01KY848DSJBK6WPEQVZY7VNCK5) |
| `2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed3` | 16 | 8 | 0.875 | 3 | [01KY848PSNWM9YH8AR5T2MZ8SR](https://beaker.org/ex/01KY848PSNWM9YH8AR5T2MZ8SR) |
| `2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed2` | 32 | 4 | 0.9375 | 2 | [01KY84900QVSTAS9GEW2E3HNGC](https://beaker.org/ex/01KY84900QVSTAS9GEW2E3HNGC) |
| `2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed3` | 32 | 4 | 0.9375 | 3 | [01KY849FHCTZGFHSCK22D6R1FD](https://beaker.org/ex/01KY849FHCTZGFHSCK22D6R1FD) |

### Launch commands

```bash
OC=true EXP=2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed2 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 \
  --max_grad_norm 1.0 --seed 2 --never_give_up 0.875 --async_steps 2

OC=true EXP=2k_ngu0875_dapo_n16_k8_gradnorm1_async2_seed3 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 16 --num_samples_per_prompt_rollout 8 \
  --max_grad_norm 1.0 --seed 3 --never_give_up 0.875 --async_steps 2

OC=true EXP=2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed2 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 32 --num_samples_per_prompt_rollout 4 \
  --max_grad_norm 1.0 --seed 2 --never_give_up 0.9375 --async_steps 2

OC=true EXP=2k_ngu09375_dapo_n32_k4_gradnorm1_async2_seed3 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 32 --num_samples_per_prompt_rollout 4 \
  --max_grad_norm 1.0 --seed 3 --never_give_up 0.9375 --async_steps 2
```

## 2026-07-24: Reinforce-Ada-Seq baseline — NGU=1.0, async_steps=1, no active_sampling (n8_k16, 3 seeds)

New baseline variant: `--never_give_up 1.0` (always retry a zero-std/
non-improving group instead of discarding it) combined with `--async_steps 1`
(minimum allowed, tightest generation/training coupling) and
`--active_sampling False` (script default is `--active_sampling` on).
`active_sampling` had to be explicitly disabled — `StreamingDataLoaderConfig.__post_init__`
(`open_instruct/data_loader.py:558-563`) asserts `async_steps > 1` whenever
`active_sampling` is on, and this combination has crashed a job before (see
the [`dapo_evalonly_n128` launch log](#2026-07-16-new-initial-model-difficulty-eval-n128-samplesprompt-replaces-w47m67sf)).
`ArgumentParserPlus` (`HfArgumentParser`) takes the *last* occurrence of a
repeated flag, so appending `--async_steps 1 --active_sampling False` after
the wrapper script's own `--async_steps 4 --active_sampling` correctly
overrides both — verified locally with a standalone `parse_args_into_dataclasses`
call before launching.

`n8_k16` (script defaults), `max_grad_norm=1.0`, `--total_episodes 256000`
(2000 steps), same image/workspace/cluster as the rest of the NGU sweep, no
rebuild needed (local diff since the image build is docs-only).

| Name | never_give_up | async_steps | active_sampling | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed1` | 1.0 | 1 | False | 1 | [01KYB4MHVH8A0QTE0ZE584MFM9](https://beaker.org/ex/01KYB4MHVH8A0QTE0ZE584MFM9) |
| `2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed2` | 1.0 | 1 | False | 2 | [01KYB4MNTKEQHF2CX0THGS748J](https://beaker.org/ex/01KYB4MNTKEQHF2CX0THGS748J) |
| `2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed3` | 1.0 | 1 | False | 3 | [01KYB4MT827BBBEDX9D4X6H29J](https://beaker.org/ex/01KYB4MT827BBBEDX9D4X6H29J) |

### Launch commands

```bash
OC=true EXP=2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --max_grad_norm 1.0 --seed 1 \
  --never_give_up 1.0 --async_steps 1 --active_sampling False

OC=true EXP=2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed2 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --max_grad_norm 1.0 --seed 2 \
  --never_give_up 1.0 --async_steps 1 --active_sampling False

OC=true EXP=2k_ngu1_dapo_n8_k16_gradnorm1_async1_noactive_seed3 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev CLUSTER=ai2/jupiter \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --max_grad_norm 1.0 --seed 3 \
  --never_give_up 1.0 --async_steps 1 --active_sampling False
```
