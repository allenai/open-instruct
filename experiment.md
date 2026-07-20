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
| `2k_ngu0875_dapo_n8_k16_gradnorm1_async2_seed1` | 8 | 16 | 0.875 | 2 | 1 | [01KX9FGGQ5QGBTXGCKESYB8C91](https://beaker.org/ex/01KX9FGGQ5QGBTXGCKESYB8C91) |

### Launch command (repeat per row; first row built the image via build_image_and_launch.sh)

```bash
OC=true EXP=2k_ngu${P}_dapo_n8_k16_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up $P --async_steps 2
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

## NGU sequence-continuation (`--ngu_seq_multiplier`): 8k×2 vs plain 16k

New feature (commit `84b617078`): with `--ngu_seq_multiplier M > 1`, a
never-give-up retry *resumes* completions that hit the `response_length` cap
instead of discarding them — the partial response is re-fed as prompt tokens
(its tokens/masks/logprobs kept as response state) and gets `response_length`
more tokens per retry, up to `response_length * M` total. Finished-but-wrong
completions still get fresh samples; continued completions are excluded from
the NGU pending buffer/baseline until their stitched version returns. M=2 at
8k ≈ a 16k budget with a halfway NGU check-in, which is exactly the
comparison: does the check-in beat just generating 16k outright?

Both arms: n=8, k=16, NGU p=0.75, `--max_grad_norm 1.0 --async_steps 2`,
seed 1, `--pack_length 18432` (2048 prompt + 16384 max response), and
`--eval_response_length 16384` so both arms get identical eval budgets.
Note this makes eval budgets *larger* than the earlier 8k-eval NGU runs, so
compare these two arms to each other, not directly to the old sweep numbers.

### Smoke test (2 GPU, Qwen3-0.6B-Base, grpo_fast backend)

Run locally (this session had 2 local L40S GPUs available) instead of on
Beaker: `uv run python open_instruct/grpo_fast.py` with the same args as
`scripts/train/debug/ngu_quartiles_2gpu.sh` (which now also takes the image
from `BEAKER_IMAGE` and forwards extra args, for the Beaker path), plus
`--ngu_seq_multiplier 2 --pack_length 3072`; `--never_give_up 1.0` + 1k
response length force lots of truncation → continuations. Original Beaker
smoke job `01KXZ13RCYYXXKXAGCVMGTM1JB` was cancelled before it started.

| Name | Notes | Where |
| --- | --- | --- |
| `ngu_seq_multiplier_local_smoke` (mult2) | commit `84b617078`, `--ngu_seq_multiplier 2 --pack_length 3072` | local, 2x L40S |

Passed on the 3rd fix iteration (first two were local-environment/pre-existing issues, not
bugs in this feature; the third was a real bug this feature introduced — see below). Final run
completed cleanly: `episode: 256/256`, model saved, `sequence_lengths_max: 2048.00` (exactly
`response_length(1024) × ngu_seq_multiplier(2)`, confirming a continuation actually reached the
multiplied cap), `stop_rate: 0.86`, no crash.

1. `LookupError: setuptools-scm was unable to detect version` — Ray's `uv_runtime_env_hook`
   auto-triggers when the driver is launched via `uv run`, shipping the working dir (`.git`
   excluded per `grpo_fast.py`'s `ray.init` runtime_env) to each worker and rebuilding the
   package there; setuptools-scm can't infer a version without `.git`. Fixed by launching with
   `.venv/bin/python` directly instead of `uv run python`, which skips the hook. Pre-existing
   local-execution-only issue, unrelated to this feature.
2. `TypeError: Object of type <class 'torch.dtype'> is not serializable` in vLLM's EngineCore
   output-socket encoder, during engine warmup before any generation. Worked around with
   `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Pre-existing vLLM/environment issue, unrelated to this
   feature.
3. **Real bug, fixed in commit `62baca0c5`**: `ValueError: Expected each prompt sample count to be
   a multiple of samples_per_prompt, got sample_count=12 and samples_per_prompt=8` in
   `expand_prompt_lengths_for_response_groups` (called from `one_training_step`'s utilization/MFU
   accounting). Root cause: NGU continuations let a merge round finalize *fewer* than
   `num_samples_per_prompt_rollout` responses (the rest are deferred, resumed into a later round),
   so a continuation-affected group's `sample_count` is no longer guaranteed to be a multiple of
   `num_samples_per_prompt_rollout` — an invariant the utilization-metrics code silently relied on
   (previously always true, since plain NGU always merges whole `k`-sized rounds). Fixed by adding
   `Group.attempt_count`/`BatchStatistics.prompt_attempt_counts` (the true generation-round count,
   independent of `sample_count`) for `prefill_flops`'s round-level accounting, and a new
   `pad_response_lengths_for_attempt_counts` helper that zero-pads each group's response lengths to
   round-align for `decode_flops`/`decode_memory_bytes`/`calculate_learner_utilization` (zero-padding
   doesn't change any FLOPs/byte/token sum, so this is exact for everything except
   `calculate_learner_utilization`'s training-mode accounting, which treats each pad slot as a small,
   bounded phantom `prompt_length`-only sequence — an observability-only approximation, not a
   training-correctness issue). All existing MFU/MBU tests (including the bit-exact
   `test_mbu_reproduction` fixtures) still pass unchanged.

**Follow-up bug found via the live Beaker run (commit `49a043644`):** `calculate_utilization_metrics`
has *two* independent call sites — `grpo_fast.py`'s `one_training_step` (DeepSpeed backend) and
`grpo_callbacks.py`'s `StepTimingCallback.post_step` (OLMo-core backend, used by `OC=true`/`grpo.py`,
which is what the experiment arms below actually run). Commit `62baca0c5` only patched the former;
the continuation arm (`01KXZWJMRC4VK0PBPYR281R72M`) crashed on the exact same `ValueError` at step
18/2000 the moment a real continuation merge occurred. Fixed by threading
`prompt_attempt_counts` through `StepTimingCallback` too. Verified against a fresh Beaker run
(`01KY05JXM1NEJWD9T9FJHW96J7`) reaching step 32/2000 cleanly, well past the prior crash point.

### Experiment arms

Workspace note: arm 1 launched under `ai2/olmo-instruct` (per
mid-session correction); arm 2 had already started under
`ai2/open-instruct-dev` before that correction landed, so it was left running
rather than restarted. Workspace is organizational only and doesn't affect
comparability.

| Name | response_length | ngu_seq_multiplier | never_give_up | Seed | Workspace | Beaker |
| --- | --- | --- | --- | --- | --- | --- |
| `2k_ngu075_mult2x8k_dapo_n8_k16_gradnorm1_async2_seed1` | 8192 | 2 | 0.75 | 1 | ai2/olmo-instruct | ~~[01KXZWJMRC4VK0PBPYR281R72M](https://beaker.org/ex/01KXZWJMRC4VK0PBPYR281R72M)~~ (crashed step 18/2000, commit `84b617078`) → ~~[01KY02BGKXQPN68PAVA2XNK4NN](https://beaker.org/ex/01KY02BGKXQPN68PAVA2XNK4NN)~~ (crashed step 18/2000 again — `grpo_callbacks.py` call site fix missing, commit `105426b07`) → [01KY05JXM1NEJWD9T9FJHW96J7](https://beaker.org/ex/01KY05JXM1NEJWD9T9FJHW96J7) (commit `49a043644`, both call sites fixed; confirmed past step 32/2000, running) |
| `2k_ngu075_seq16k_dapo_n8_k16_gradnorm1_async2_seed1` | 16384 | 1 | 0.75 | 1 | ai2/open-instruct-dev | [01KXZ3J985XXE89MTGG2BQSTXF](https://beaker.org/ex/01KXZ3J985XXE89MTGG2BQSTXF) |

### Launch commands

```bash
# Arm 1: continuation (8k chunks, up to 16k via NGU retries)
OC=true EXP=2k_ngu075_mult2x8k_dapo_n8_k16_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/olmo-instruct \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2 \
  --ngu_seq_multiplier 2 --pack_length 18432 --eval_response_length 16384

# Arm 2: plain 16k baseline (same NGU p, no continuation)
OC=true EXP=2k_ngu075_seq16k_dapo_n8_k16_gradnorm1_async2_seed1 \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/open-instruct-dev \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --never_give_up 0.75 --async_steps 2 \
  --response_length 16384 --pack_length 18432
```

## Smoke test (2 GPU, before launching the sweep)

Quick NGU + per-quartile-metrics check on a small model via
`scripts/train/debug/ngu_quartiles_2gpu.sh`.

| Name | Notes | Beaker |
| --- | --- | --- |
| `ngu_quartiles_2gpu` | 2 GPU, Qwen3-0.6B-Base, 256 episodes, `--active_sampling --never_give_up 1.0` | [01KW2XH2WYC158J2ESK4S1F3TY](https://beaker.org/ex/01KW2XH2WYC158J2ESK4S1F3TY) |
