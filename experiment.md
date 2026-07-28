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
| `2k_baseline_dapo_n8_k16_gradnorm1_async2_seed1_16k` | 16384 | 1 | 0 (no NGU) | 1 | ai2/olmo-instruct | ~~[01KY0BC34QVCHPPWXBE3GDCZF9](https://beaker.org/ex/01KY0BC34QVCHPPWXBE3GDCZF9)~~ (CUDA OOM step 290/2000, `torch.OutOfMemoryError` in `feed_forward.w1`, 76.03/79.19 GiB allocated) → ~~[01KY0R82P4PVPAADY1HBFPRWEW](https://beaker.org/ex/01KY0R82P4PVPAADY1HBFPRWEW)~~ (`--activation_memory_budget 0.25`; crashed in ~2min, unrelated `torch.distributed.DistNetworkError: EADDRINUSE` during vLLM engine startup on `jupiter-cs-aus-138` — transient port-collision infra flake, not a memory issue, different node than the OOM run) → [01KY0T45SY5EF75X6PWC617340](https://beaker.org/ex/01KY0T45SY5EF75X6PWC617340) (clean relaunch, same `--activation_memory_budget 0.25`; **fix confirmed** — passed step 290 cleanly (reached step 305+ with no OOM/errors over 2+ hours), running) |

Third arm added to isolate whether NGU (in either form) helps at all relative
to no-revisit at the same 16k sequence-length ceiling.

**Arm 3 OOM root-cause note (2026-07-20):** crashed at step 290/2000 with
76.03/79.19 GiB allocated — genuine memory pressure, not a bug.
`--gradient_checkpointing` on this command line is a no-op on the OLMo-core
(`OC=true`/`grpo.py`) path — it only wires into `grpo_fast.py`'s DeepSpeed
backend (see `model_utils.py`'s `gradient_checkpointing` field, never read by
`grpo_olmo_core_actor.py`/`olmo_core_train_modules.py`). The real
activation-checkpointing knob for OLMo-core is `--activation_memory_budget`
(needs `--compile_model` default `True`; see `build_ac_config` in
`olmo_core_utils.py`), which was already 0.5 here but with `fsdp_shard_degree
4` (no extra sharding headroom) and `--pack_length 18432` that left too
little margin. Sibling arm 2 (NGU, same 16k ceiling) ran past step 500
without issue, so this looks specific to the no-NGU arm — plausibly because
without NGU retries more completions run close to the full pack length.
Considered but ruled out: PR #1747's tiled GRPO loss (`--use_liger_grpo_loss`)
only touches `grpo_fast.py`/`grpo_utils.py`, not the OLMo-core actor, so it
doesn't apply here. Fix: relaunched with `--activation_memory_budget 0.25`
(precedented elsewhere for long-context OC=true GRPO, e.g.
`multi_node_grpo.sh` uses 0.25 with pack_length 20480, albeit with more
learner-GPU sharding). Not resumed from the step200 checkpoint — relaunched
fresh from step 0, consistent with how prior crash-relaunches in this file
were handled. **Confirmed fixed:** relaunch `01KY0T45SY5EF75X6PWC617340` ran
past step 290 cleanly (reached step 305+ over 2+ hours wall clock, no
OOM/errors).

**New observation while confirming the fix (2026-07-21):** step throughput on
this relaunch dropped sharply around the same step range — ~1.9 steps/min
(steps 217→293) down to ~0.16 steps/min (steps 301→305), ETA growing from
~8h to ~12h. In-loop eval logs show `sequence_lengths` mean jumping
1575→7005 tokens and `stop_rate` dropping 0.97→0.81 between the 23:47 and
01:03 eval prints. This is the same signature as the `val/rho_weight`
completion-length-drift collapse documented in the [rho_weight collapse
entry](research.md#rho_weight-collapse-under-grad_norm10-n4_k32-watch-n2_k64-too-root-cause--partial-fix)
— worth watching whether this arm stalls the way `n4_k32_gradnorm1` did.
Not yet confirmed as the same collapse (no direct `val/rho_weight` line in
stdout logs to check, only inferred from sequence-length/stop-rate proxies);
flagging for follow-up, not treating as resolved.

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

# Arm 3: no-NGU baseline, same 16k budget
OC=true EXP=2k_baseline_dapo_n8_k16_gradnorm1_async2_seed1_16k \
  BEAKER_IMAGE=michaeln/open-instruct-integration-test-ngu WORKSPACE=ai2/olmo-instruct \
  bash scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --total_episodes 256000 --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 16 \
  --max_grad_norm 1.0 --seed 1 --async_steps 2 \
  --response_length 16384 --pack_length 18432
```

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

## 2026-07-24: reinforce_ada_est implementation + 3-seed launch (grpo.py, OC)

New feature `--reinforce_ada_est` (see
[research.md](research.md#active-reinforce_ada_est-adaptive-completions-per-prompt-from-pre-computed-pass_count)):
derives each prompt's completions-per-rollout from its `pass_count` column
(correct-out-of-32) instead of a fixed `num_samples_per_prompt_rollout` for
every prompt: pass_count >= 8 -> 4, >= 4 -> 8, >= 2 -> 16, else -> 32.

Code changes:
- `open_instruct/dataset_transformation.py`: `PASS_COUNT_KEY = "pass_count"` constant. No transform-pipeline change needed — `grpo_fast.setup_datasets` never sets `DatasetConfig.target_columns`, so raw dataset columns already survive tokenization untouched (same mechanism that already preserves `tools`/`env_config`).
- `open_instruct/data_loader_utils.py`: `compute_reinforce_ada_est_samples(pass_count) -> int` bucketing helper.
- `open_instruct/data_loader.py`: new `StreamingDataLoaderConfig.reinforce_ada_est: bool` field (validated: requires `batch_by="prompts"`, incompatible with `never_give_up`); `add_prompt_to_generator` overrides the per-request `generation_config.n` via `dataclasses.replace` when set (skipped for eval prompts); `process_group`'s response-count assert checks the bucketed count instead of the global `generation_config.n`. `accumulate_inference_batches`/`maybe_replenish_prompt` thread the flag through. No changes needed in `compute_grouped_advantages`/`expand_grouped_scores` (already group by a per-prompt `sample_count` list, not a fixed scalar) or in vLLM request handling (`vllm_utils.add_request` already loops `range(request.generation_config.n)` per request).
- `open_instruct/grpo_fast.py`: `setup_datasets` raises a clear error if `reinforce_ada_est` is set but the train dataset has no `pass_count` column. (`open_instruct/grpo.py` calls this same `setup_datasets`, so the check covers both entry points; only `grpo.py`/OC=true is actually being launched here per explicit request.)
- `open_instruct/test_data_loader.py`: bucket boundary tests, config validation tests, `add_prompt_to_generator`/`process_group` behavior tests (12 new tests, `TestReinforceAdaEst`). Full `test_data_loader.py` suite (31 tests) and `make style && make quality` both pass.
- New debug script `scripts/train/debug/reinforce_ada_est_2gpu.sh` (mirrors the existing `ngu_quartiles_2gpu.sh` smoke test, but calls `open_instruct/grpo.py` instead of `grpo_fast.py`).

Known approximation: batch-size/pool/queue-sizing and episode-count math elsewhere (`grpo.py`/`grpo_fast.py`/`data_loader.py`) still use the nominal `num_samples_per_prompt_rollout` CLI value as an average for sizing/logging purposes — not exact given the variable per-prompt `n`, but this is the same pre-existing approximation NGU's variable group sizes already rely on (see `expand_prompt_lengths_for_response_groups`/`calculate_utilization_metrics`, which already consume the actual per-group `prompt_sample_counts` where it matters for correctness — MFU/utilization metrics — leaving only coarse estimates like `num_training_steps`/pool sizing on the nominal value).

### Launch

Smoke test (2 GPU, `open_instruct/grpo.py`, `Qwen3-0.6B-Base`, 256 episodes):
```
./scripts/train/build_image_and_launch.sh scripts/train/debug/reinforce_ada_est_2gpu.sh
```

3-seed production launch (`open_instruct/grpo.py`/OC=true, otherwise identical to `scripts/train/qwen/qwen3_4b_deepscaler_math.sh` defaults):
```
OC=true EXP=reinforce_ada_est_seed<N> ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh --reinforce_ada_est True --seed <N>
```
for `<N>` in 1, 2, 3.

Beaker links and outcome:

- Local smoke test debugging (ran directly on 2 local GPUs, not Beaker, since the
  Beaker `--priority urgent` queue was congested/stuck): found and fixed a latent
  bug in `reinforce_ada_est_2gpu.sh` (and its precedent `ngu_quartiles_2gpu.sh`,
  left unfixed there as out of scope) — with a single learner GPU
  (`num_learners_per_node 1`, world_size 1) and `single_gpu_mode` left at its
  default `False`, `grpo_olmo_core_actor.py` skips building `dp_config`
  (`not single_gpu_mode and world_size > 1`) and therefore never casts the model
  to bf16, crashing FlashAttention with `RuntimeError: FlashAttention only
  support fp16 and bf16 data type` on the dry-run batch. Fixed by adding
  `--single_gpu_mode True` to `reinforce_ada_est_2gpu.sh` (commit `871ee88f7`).
  Not a `reinforce_ada_est` bug — the production 3-seed launch uses
  `fsdp_shard_degree 4`/`num_learners_per_node 4` (world_size 4), so `dp_config`
  is built normally there regardless of this flag.
- After the fix, local smoke test completed both training steps end-to-end
  with `--reinforce_ada_est True` on `open_instruct/grpo.py`: prompt-level
  `pass_count` correctly drove per-request `n` in `add_prompt_to_generator`,
  `process_group` accepted the bucketed response counts with no assert
  failures, `accumulate_inference_batches` correctly handled the variable
  group sizes under `active_sampling`, and `[step=1/2,epoch=1]` /
  `[step=2/2,epoch=1]` both logged followed by `Training complete`.
- `uv run pytest open_instruct/test_grpo_fast.py -q`: 22 passed, 1 skipped, no
  failures (confirms no regression in the shared `grpo_fast.py`/
  `accumulate_inference_batches` coverage).
- 3-seed production launch, all `open_instruct/grpo.py` (OC=true),
  `--reinforce_ada_est True`, image built from commit `871ee88f7`:
  1. seed 1: [Beaker](https://beaker.org/ex/01KYBC5WT2F0G2X4SGS0YKXDYD)
  2. seed 2: [Beaker](https://beaker.org/ex/01KYBC73KRNTEWZ6JZNTJ39RAN)
  3. seed 3: [Beaker](https://beaker.org/ex/01KYBCBTWH74DZRZ7HWAQBN5K9) (first
     attempt `01KYBC8AM8PT1V6X1SEA15PSE2` was auto-canceled ~1 min after
     scheduling: node `01KQ0MFSCB5MJE2HZ3CVBAHTQJ` was cordoned for an
     unrecoverable SXid GPU error, unrelated to this change; relaunched.)

  All three jobs were kicked off successfully by mason.py (`ai2/jupiter`,
  8 GPUs each, `--priority urgent`, `--preemptible`) and confirmed `scheduled`
  (not stuck in `created`). Training-progress/convergence outcome still TBD,
  to be checked and recorded once the runs have made meaningful progress.

**Update 2026-07-25: all 3 seeds crashed within ~1-3 min (`exitCode=1`),
fix + relaunch.** All three (`01KYBC5WT2F0G2X4SGS0YKXDYD`,
`01KYBC73KRNTEWZ6JZNTJ39RAN`, `01KYBCBTWH74DZRZ7HWAQBN5K9`) finalized around
2026-07-25 00:59-01:02 with:

```
ValueError: Group sample_count=32 exceeds attempt_count(1) * samples_per_prompt(16);
attempt_count must cover every sample.
```
raised in `pad_response_lengths_for_attempt_counts` (`open_instruct/utils.py`),
called from `StepTimingCallback.post_step` (`open_instruct/grpo_callbacks.py`)
-> `utils.calculate_utilization_metrics`. Root cause: this helper's negative-
`pad_count` guard assumed `sample_count` only ever grows via `never_give_up`'s
multi-round merge (which increments `attempt_count` in lockstep), but
`reinforce_ada_est` also changes a group's `sample_count` away from the
baseline `samples_per_prompt` (bucketing a `pass_count`-0/1 prompt to 32
samples when `--num_samples_per_prompt_rollout 16`) via the
`never_give_up == 0` code path in `maybe_filter_group`
(`open_instruct/data_loader.py`), which never touches `attempt_count` at all
(stays at the dataclass default `1`). So `32 > 1*16` deterministically raises
on the very first low-`pass_count` prompt in a batch -- this is a
guaranteed-to-trigger bug for `reinforce_ada_est`, not an intermittent race.

Fix (commit `f26cba8ab`): in `calculate_utilization_metrics`, correct
`prompt_attempt_counts` up to `ceil(sample_count / samples_per_prompt)`
before both `expand_prompt_lengths_for_response_groups` and
`pad_response_lengths_for_attempt_counts` consume it, so their alignment
invariant (prompt-length attribution shifting for every later group in the
batch if left uncorrected) holds regardless of which feature changed
`sample_count`. Also made `pad_response_lengths_for_attempt_counts` warn-and-
skip padding instead of raise, as a defense-in-depth safety net (this
function is explicitly documented as observability-only, not
correctness-critical for training). Added
`test_utilization_metrics_handles_under_counted_attempt_count` and updated
`test_pad_response_lengths_for_attempt_counts_raises_on_impossible_attempt_count`
(renamed to `..._skips_padding_...`, no longer expects a raise) in
`open_instruct/test_utils.py`. `make style && make quality` and
`uv run pytest open_instruct/test_utils.py -k "pad_response_lengths or utilization_metrics"`
both pass (5/5); full `test_utils.py` has 6 pre-existing unrelated
`CombineDatasetTest` failures confirmed present on `main` too (not caused by
this change).

Relaunched all 3 seeds from the fixed commit (`ai2/oe-adapt-code` workspace,
`--priority high` per request), same config otherwise
(`--reinforce_ada_est True`, `open_instruct/grpo.py`, OC=true):

1. seed 1: [Beaker](https://beaker.org/ex/01KYDA56FVZYR0GKRAZCCE44AX)
2. seed 2: [Beaker](https://beaker.org/ex/01KYDA5Q7QAYSQRMJP6RKPDZ3B)
3. seed 3: [Beaker](https://beaker.org/ex/01KYDA66PXAZ6FN53GH9X4KR4T)

(Note: first launch attempt accidentally omitted `OC=true` and ran on
`grpo_fast.py` -- `01KYDA45NWE8WRJV07R5H33R6A` -- caught immediately and
stopped before it made progress; not a real run.)

All three confirmed `started` shortly after launch. Training reached
`training_step` 200+ on seeds 1/2 with no recurrence of the
`attempt_count`/`pad_response_lengths_for_attempt_counts` crash, and seed 1's
step-100 checkpoint + eval both completed cleanly (`eval scores: 1.46`),
confirming the fix holds through the checkpoint/eval code path too.

**seed 3 crashed at `training_step` ~200 (2026-07-25 19:49), unrelated
infra failure.** `RuntimeError: Application timeout caused pair closure` in
`torch.distributed.all_reduce` (`olmo_core/train/utils.py:320`, called from
`StepTimingCallback`'s metrics reduction) -- a distributed collective-ops
network timeout, not the `attempt_count` bug (no `pad_response_lengths_for_attempt_counts`
signature in the traceback at all). Ran cleanly for ~53 minutes before
failing, no `canceledFor`/`canceledCode` (unlike the earlier cordoned-node
case), so likely a transient network blip between ranks during an
all-reduce rather than a code or hardware fault. Relaunched with the
identical command:

```
OC=true WORKSPACE=ai2/oe-adapt-code PRIORITY=high EXP=reinforce_ada_est_fixed_seed3 \
  ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --reinforce_ada_est True --seed 3
```

seed 3 (relaunch): [Beaker](https://beaker.org/ex/01KYDDJ1MWXAEV3RSABT6BYGC6)

**seed 3's relaunch crashed again at `training_step=202` (2026-07-25 20:46),
same failure signature**: `RuntimeError: [gloo/transport/tcp/pair.cc:547]
Connection closed by peer` / `Application timeout caused pair closure`.
Confirmed this is *not* the same bad node both times -- first crash was on
`jupiter-cs-aus-120` (`10.93.1.29`), second on `jupiter-cs-aus-133`
(`10.93.1.42`) -- and seeds 1/2 were already well past `training_step`
200-280 with no issue at the time of both crashes, ruling out a systemic
bug tied to the `reinforce_ada_est` fix or feature (would expect seeds 1/2
to hit it too if so). Genuine, if unlucky (2/2), transient network
flakiness on the `--preemptible` GPU pool. Relaunched a third time,
identical command:

seed 3 (2nd relaunch): [Beaker](https://beaker.org/ex/01KYDGJHG34KDJT80W88SH4HHM)

**seed 1 also crashed, same signature, at `training_step=322` (2026-07-25
21:36)**: `RuntimeError: [gloo/transport/tcp/unbound_buffer.cc:78] Timed out
waiting 1800000ms for recv operation to complete` -> `Application timeout
caused pair closure`. This is the third occurrence of the identical Gloo
collective-timeout signature in ~2.5 hours, now across 2 of the 3 seeds.
Checked `DataPreparationActor.get_data` logs right up to the crash: step
cadence was normal (no rank stuck at a high `wait_count`, steps advancing
every ~10-30s through step 321) -- no evidence of an application-level
stall or a `reinforce_ada_est`-specific hang, which continues to point at
`ai2/jupiter` network flakiness rather than a code issue (seed 2 was
already well past this step range throughout, unaffected).

**Lesson learned, mid-course correction:** the first two seed-3 relaunches
above used the plain launch command, which -- via `mason.py`'s
`--auto_checkpoint_state_dir` -- assigns a *fresh* `--checkpoint_state_dir`
each time, discarding the crashed run's saved checkpoints and restarting
from step 0. Confirmed via `ls` that seed 1's original checkpoint dir
(`.../deletable_checkpoint_states/michaeln/1785005775_707439`) had a
`step300` checkpoint saved ~30 min before its crash. `mason.py` skips the
auto-override if `--checkpoint_state_dir` is already explicitly set to a
`/weka/`-prefixed path, so relaunched seed 1 with that flag pointed at the
existing checkpoint dir to resume from `step300` instead of losing ~2 hours
of progress:

```
OC=true WORKSPACE=ai2/oe-adapt-code PRIORITY=high EXP=reinforce_ada_est_fixed_seed1_resume \
  ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --reinforce_ada_est True --seed 1 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1785005775_707439
```

seed 1 (resumed from step300): [Beaker](https://beaker.org/ex/01KYDKNJ9TSPTRXYQ1Q4S9K7X3)
(a first relaunch attempt without the checkpoint override,
`01KYDKK3JY1JNMQ452A3FV04JV`, was caught and stopped before making progress
-- not a real run.)

Seed 3's two prior relaunches did *not* get this treatment (each restarted
from step 0) since this was only noticed at seed 1's crash.

**seed 3's 3rd attempt also crashed, identical signature, at
`training_step` ~330-ish (2026-07-25 23:22-23:23)**: same
`Application timeout caused pair closure` / gloo `unbound_buffer.cc`
1800000ms timeout. This is the 4th occurrence of this exact signature in
~4 hours -- 3 of the 4 have now hit seed 3 specifically (on 3 different
nodes/IPs each time, ruling out one bad node), vs. 1 for seed 1 and 0 for
seed 2. Resumed from the checkpoint dir noted above (had a `step300`
checkpoint saved ~30 min before this crash):

```
OC=true WORKSPACE=ai2/oe-adapt-code PRIORITY=high EXP=reinforce_ada_est_fixed_seed3_resume \
  ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_4b_deepscaler_math.sh \
  --reinforce_ada_est True --seed 3 \
  --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1785012503_30650
```

seed 3 (resumed from step300, 4th attempt): [Beaker](https://beaker.org/ex/01KYDSHYB8ZKV2SFT31MNG6M3T)

At this point (4 crashes, same exact signature, disproportionately on
seed 3) this looks less like isolated bad luck and more like either a
genuine `ai2/jupiter`-wide network instability window, or something about
this specific job's placement/resource profile that keeps triggering
30-min collective timeouts -- worth raising with infra/other users of the
cluster if it keeps recurring, rather than continuing to blindly relaunch
indefinitely.

**Separately, seed 1's resumed job (`01KYDKNJ9TSPTRXYQ1Q4S9K7X3`) was
preempted** (2026-07-26 00:21) by an unrelated higher-priority `urgent` job
elsewhere on the shared cluster (`canceledFor`: "preempted by job ... with
'urgent' priority") -- not a crash, and unrelated to any of the above. The
job spec has `autoResume: true`, so Beaker automatically scheduled and
started a replacement job within the same experiment a few minutes later,
no manual relaunch needed. Worth noting for future launches: our jobs run
at `--priority high`, so they're preemptible by any `urgent`-priority job
on `ai2/jupiter`, independent of the gloo-timeout pattern above.

**seed 3's 4th attempt (resumed) crashed again, identical signature, at
`training_step` ~430ish (2026-07-26 01:15-01:16)**: 5th occurrence of the
exact same `Application timeout caused pair closure` / gloo
`unbound_buffer.cc` 1800000ms timeout in under 6 hours, now 4 of 5 on
seed 3 specifically (vs. 1 for seed 1, 0 for seed 2, which has meanwhile
progressed past `training_step` 500+ without any issue). Resumed again
from the `step400` checkpoint in the same dir:

seed 3 (resumed from step400, 5th attempt): [Beaker](https://beaker.org/ex/01KYE04NF9502KDCY2NK5HQJAD)

This concentration on one seed's jobs specifically (not a cluster-wide
effect that would hit all 3 roughly evenly) is now the strongest signal
yet that something about this seed's specific job/placement is implicated,
not pure bad luck -- flagged directly to the user rather than continuing
to auto-relaunch indefinitely without visibility into cluster-side causes.

**Update: seed 2 also crashed with the identical signature at
`training_step=721` (2026-07-26 04:18), after running rock-solid for
~700 steps.** This revises the "seed-3-specific" theory above: seed 2 had
been the least-affected job by far right up until this point, so the
crash isn't concentrated on one seed's placement -- it's better explained
as a genuine `ai2/jupiter`-wide network flakiness with some baseline
per-job-hour failure rate, and seed 3's higher crash *count* is mostly an
artifact of it having accumulated more separate job-hours (via repeated
restarts) than a real per-hour rate difference. 6th occurrence of the
identical signature overall. Resumed from its `step700` checkpoint:

seed 2 (resumed from step700): [Beaker](https://beaker.org/ex/01KYEAKKGA6N65GM69AQ0WBB62)

Training-progress/convergence outcome still TBD.

## 2026-07-25: DeepCoder-1.5B data pipeline + K/NGU sweep launch (grpo.py, OC)

Picked up a handoff from another session/machine that had written
`scripts/data/create_deepcoder_data.py` (converts
`agentica-org/DeepCoder-Preview-Dataset` into RLVR `code_stdio` format,
pushing `mnoukhov/deepcoder_{lcbv5,lcbv5_test,primeintellect,taco,codeforces_test}`)
and `scripts/train/qwen/deepcoder_1_5b.sh` (GRPO on
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `open_instruct/grpo.py`/OLMo-core
only) but had never run either to completion. The handoff's branch state
didn't match this machine (stale local `ngu`, diverged from `origin/ngu`
where the work actually lived) — resolved via `git fetch` + merge (commit
`093117162`), reconciling with unrelated `reinforce_ada_est` work done in
parallel on this machine.

### Data conversion, take 1: Arrow overflow

First full run of `create_deepcoder_data.py` (directly on this machine —
2TB RAM, not the 30GB box the handoff described) pushed all 5 datasets, but
`lcbv5` crashed on write: `pyarrow.lib.ArrowInvalid: offset overflow while
concatenating arrays`. Root cause: a handful of LiveCodeBench-v5
stress-test problems ship up to 30 tests with individual test cases running
several MB each (worst case: one problem's `ground_truth` was 160MB, lcbv5's
322-row total was 4.79GB), overflowing PyArrow's 2GB string-offset limit
during dataset caching.

This isn't just a storage problem — `CodeStdioVerifier.async_call`
(`open_instruct/ground_truth_utils.py`) bundles *every* test for one example
into a single HTTP POST to the code-execution API, an AWS API Gateway
endpoint (10MB hard request limit, Lambda sync-invoke caps ~6MB). Multi-MB
tests would fail at grading time regardless of the Arrow issue.

Fixed in `create_deepcoder_data.py` (`cap_tests`): drop any single test
over 100KB, greedily keep the largest ("most challenging", matching
DeepCoder's own recipe of sampling "the 15 most challenging tests" per
problem — https://www.together.ai/blog/deepcoder) remaining tests within a
500KB total budget per problem. Verified against the already-pushed
(uncapped) data: only 3 of 21,707 problems drop to zero usable tests, median
test count lands at 12–15 (matches the reference recipe), regenerated
datasets top out at 726KB/example (~8x margin under the 6MB Lambda ceiling).
Re-ran and re-pushed all 5 datasets; no Arrow errors. Final counts:

| Dataset | Split | Count |
| --- | --- | --- |
| `mnoukhov/deepcoder_lcbv5` | train | 322 |
| `mnoukhov/deepcoder_lcbv5_test` | eval | 175 |
| `mnoukhov/deepcoder_primeintellect` | train | 14995 (3 dropped, zero tests under cap) |
| `mnoukhov/deepcoder_taco` | train | 6387 |
| `mnoukhov/deepcoder_codeforces_test` | eval | 408 |

### Hyperparameter decisions (resolved against the real DeepCoder/DeepScaleR training scripts)

The handoff flagged several hyperparameter choices as unconfirmed guesses.
Verified against the actual public `agentica-project/rllm` training scripts
(DeepScaleR-1.5B is the direct math-domain precursor to DeepCoder-1.5B, same
base checkpoint/team) rather than guessing:

- `--beta 0.001`: confirmed — real 1.5B script (`run_deepscaler_1.5b_24k.sh`)
  sets `kl_loss_coef=0.001` with `use_kl_loss=True` (unlike the 14B DeepCoder
  recipe, which disables KL loss entirely).
- `--num_samples_per_prompt_rollout 16`: confirmed exact match — real 1.5B
  script uses `rollout.n=16` (the handoff's "wandb showed n=8" was from the
  14B recipe, not the 1.5B one it should've been compared against).
- `--response_length 32768` / `--pack_length 34816`: deliberate override of
  the real 1.5B run's `24576` — kept as user's explicit choice.
- `--clip_higher 0.272`: kept the repo's DAPO default (matches
  `qwen3_4b_dapo_math_32k.sh`) over switching to the 1.5B run's symmetric
  ~0.2 (no `clip_ratio_high` override there) — DAPO's asymmetric clip-higher
  is close to the real 14B recipe's `clip_ratio_high=0.28` and is generally
  an improvement over symmetric clipping. User confirmed keep-as-is.
  Full comparison table:

  | Param | Script (ours) | Real 1.5B (DeepScaleR) | Real 14B (DeepCoder) |
  | --- | --- | --- | --- |
  | `beta`/`kl_coef` | 0.001 | 0.001 (active) | 0.0 (disabled) |
  | samples/prompt (`n`) | 16 | 16 | 8 |
  | response_length | 32768 | 24576 | 16k→32k (staged) |
  | clip_ratio | 0.272 (asym) | ~0.2 (symmetric, no override) | 0.2/0.28 (asym) |

- `--model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`: confirmed
  correct base checkpoint for both the DeepCoder-1.5B and DeepScaleR-1.5B
  lineages.
- `--total_episodes`: changed from the handoff's placeholder `128000` to
  `64000` per direct request.
- `WORKSPACE`: changed from `ai2/oe-adapt-code` to `ai2/olmo-instruct` per
  direct request (used for both the sanity-check launch and the sweep below).

### Sanity-check launch: three real bugs found and fixed

Per-repo convention, sanity-checked `deepcoder_1_5b.sh` before trusting the
untested 2-node topology (8 learners on node0 / 8 vLLM engines on node1,
`fsdp_shard_degree 4`/`fsdp_num_replicas 1` — inherited unmodified from the
`qwen3_4b_deepscaler_math.sh` template despite `num_learners_per_node` being
bumped to 8). User then asked to switch to a single-node topology
(`NODES=1`, 4 learners + 4 vLLM engines, mirroring the tested
`qwen3_4b_dapo_math_oc.sh` single-node OC pattern) and `--async_steps 2`
before the first real launch attempt. Five launch attempts were needed to
get a genuinely clean training step, surfacing three separate real bugs:

1. **`fsdp_shard_degree`/`num_learners_per_node` mismatch** (commit
   `ec8c8bbc6`): `grpo_utils.py` requires
   `fsdp_shard_degree * fsdp_num_replicas == total learner GPUs`; the
   template's `--fsdp_shard_degree 4` didn't match the bumped
   `--num_learners_per_node 8` (then `4` after the single-node switch).
2. **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` unsupported by
   `open_instruct/grpo.py`'s OLMo-core backend** (commits `eca28b1d1`,
   `aa442a161`): `olmo_core_utils.OLMO_MODEL_CONFIG_MAP` only ships
   OLMo-2/3 and Qwen3 presets. The checkpoint is Qwen2.5-architecture
   (HF `model_type="qwen2"`) — not just missing from the list but
   architecturally incompatible with any existing preset (no QK-norm unlike
   Qwen3, but does use QKV attention bias, which Qwen3 doesn't). Added a
   `_deepseek_r1_distill_qwen_1_5b` preset built directly from the HF
   `config.json`, plus:
   - a `"qwen2"` entry in olmo-core's `MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_*`
     mapping dicts (HF→olmo-core direction) — the generic fallback collapsed
     `input_layernorm`/`post_attention_layernorm` onto the same key and had
     no bias mapping at all;
   - a mirrored `"qwen2"` entry in
     `MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS` (reverse
     direction, used by `verify_can_save_as_hf`/`save_state_dict_as_hf`);
   - `TransformerConfig.freeze_params` pinning the synthetic `w_out.bias`
     (Qwen2 has no o_proj bias; olmo-core's `AttentionConfig.bias` applies
     uniformly to all four projections) at exactly zero for the model's
     entire lifetime, and `drop_frozen_zero_bias_for_hf_export` stripping it
     before HF export.

   Verified with a full HF→olmo-core→HF round trip: bit-exact logits/weights
   across all 339 params, both via a standalone script and through the
   actual production code path (`get_transformer_config` +
   `load_hf_model` + `verify_can_save_as_hf`).
3. **vLLM weight sync had no qwen2 bias mapping either** (commit
   `ac3f757f6`): the direct NCCL weight-broadcast path
   (`grpo_olmo_core_actor.run_initial_weight_sync` /
   `VLLMWeightSyncCallback`) uses a *third*, separate hardcoded name-mapping
   table (`grpo_callbacks._OLMO_CORE_TO_HF_LAYER_MAPPINGS`, independent of
   `olmo_core.nn.hf.convert`) with no q/k/v bias entries — those params
   passed through unmapped under their raw olmo-core name, crashing every
   vLLM engine (`There is no module or parameter named 'blocks' in
   Qwen2ForCausalLM`). Added the bias entries; `olmo_core_to_hf_name` now
   returns `None` for the synthetic `w_out.bias` (no valid destination
   either way) and callers (`_mapped_named_parameters`, new) drop it rather
   than send an invalid name.

All three fixes have offline (no-GPU-download) unit tests:
`open_instruct/test_olmo_core_finetune.py::DeepSeekR1DistillQwenConfigTest`,
`open_instruct/test_grpo_callbacks.py::OlmoCoreToHfNameTest`,
`open_instruct/test_vllm_utils.py::TestMappedNamedParameters`. `make style
&& make quality` clean throughout.

### Sweep launch

Per direct request: single seed each for K∈{16,32,64} baselines (holding
`N*K=128`, mirroring the earlier deepscaler K-ablation convention) plus NGU
`p`∈{0.5,0.75,0.875} at the K=16 baseline config, `WORKSPACE=ai2/olmo-instruct`,
`NODES=1` (4 learners + 4 vLLM engines), all built from commit `ac3f757f6`.

| Name | N | K | never_give_up | Beaker |
| --- | --- | --- | --- | --- |
| `deepcoder_1_5b_baseline_n8_k16` | 8 | 16 | — | [01KYC592P25ZQMZ8KKEYFNTX1R](https://beaker.org/ex/01KYC592P25ZQMZ8KKEYFNTX1R) |
| `deepcoder_1_5b_baseline_n4_k32` | 4 | 32 | — | [01KYC5Y6MS8EX03SX9C58H7CV9](https://beaker.org/ex/01KYC5Y6MS8EX03SX9C58H7CV9) |
| `deepcoder_1_5b_baseline_n2_k64` | 2 | 64 | — | [01KYC5YSHSVT5F2Q29C1W6AVGR](https://beaker.org/ex/01KYC5YSHSVT5F2Q29C1W6AVGR) |
| `deepcoder_1_5b_ngu05_n8_k16` | 8 | 16 | 0.5 | [01KYC5ZAD2YFA5YKR43GTFC0Z5](https://beaker.org/ex/01KYC5ZAD2YFA5YKR43GTFC0Z5) |
| `deepcoder_1_5b_ngu075_n8_k16` | 8 | 16 | 0.75 | [01KYC5ZW352ZPPDDKX74NDZP2Z](https://beaker.org/ex/01KYC5ZW352ZPPDDKX74NDZP2Z) |
| `deepcoder_1_5b_ngu0875_n8_k16` | 8 | 16 | 0.875 | [01KYC60DGG12F3VX2YEHD9K5YT](https://beaker.org/ex/01KYC60DGG12F3VX2YEHD9K5YT) |

### Launch commands

```bash
EXP=baseline_n8_k16 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh

EXP=baseline_n4_k32 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32

EXP=baseline_n2_k64 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 2 --num_samples_per_prompt_rollout 64

EXP=ngu05_n8_k16 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.5

EXP=ngu075_n8_k16 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.75

EXP=ngu0875_n8_k16 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.875
```

All 6 jobs confirmed training cleanly (each reached a logged `[step=N/500]`
within its first couple minutes, no errors) as of this writing.
`baseline_n8_k16` was watched furthest: `[step=2/500,epoch=1,eta=9h57m]`.
Convergence/reward-curve findings still TBD, to be recorded once the runs
have made meaningful progress.

### `--eval_temperature` + 2 more seeds per config

Added `--eval_temperature` (`grpo_utils.GRPOExperimentConfig`, mirrors the
existing `--eval_top_p` pattern: `None` falls back to training temperature)
so eval sampling temperature can be set independently of train temperature.
`create_generation_configs` (`grpo_fast.py`) now applies it when building the
eval `SamplingConfig`. Set `--eval_temperature 0.6` as the new default in
`deepcoder_1_5b.sh` (commit `37327f2e4`).

Launched 2 more seeds (2, 3) of each of the 6 sweep configs above — single
seed was only for the initial sanity pass; per-arm variance needs ≥3 seeds
to be meaningful, matching the DeepScaleR NGU sweep's seed convention.
Per direct request, moved these off `ai2/olmo-instruct` to
`WORKSPACE=ai2/oe-adapt-code`, `PRIORITY=high`.

| Name | N | K | never_give_up | Seed | Beaker |
| --- | --- | --- | --- | --- | --- |
| `deepcoder_1_5b_baseline_n8_k16_seed2` | 8 | 16 | — | 2 | [01KYD3T36SKVAC019XPKFV897A](https://beaker.org/ex/01KYD3T36SKVAC019XPKFV897A) |
| `deepcoder_1_5b_baseline_n8_k16_seed3` | 8 | 16 | — | 3 | [01KYD3TKHDT08DGFWQQPS5FY5E](https://beaker.org/ex/01KYD3TKHDT08DGFWQQPS5FY5E) |
| `deepcoder_1_5b_baseline_n4_k32_seed2` | 4 | 32 | — | 2 | [01KYD3V33KJVMB2H65HWHFT9MV](https://beaker.org/ex/01KYD3V33KJVMB2H65HWHFT9MV) |
| `deepcoder_1_5b_baseline_n4_k32_seed3` | 4 | 32 | — | 3 | [01KYD3VKD1DM47487MCS3M1XT0](https://beaker.org/ex/01KYD3VKD1DM47487MCS3M1XT0) |
| `deepcoder_1_5b_baseline_n2_k64_seed2` | 2 | 64 | — | 2 | [01KYD3W3NTGVDX21MCDF138GR2](https://beaker.org/ex/01KYD3W3NTGVDX21MCDF138GR2) |
| `deepcoder_1_5b_baseline_n2_k64_seed3` | 2 | 64 | — | 3 | [01KYD3WMK4B0VPYR90VKFP4VN0](https://beaker.org/ex/01KYD3WMK4B0VPYR90VKFP4VN0) |
| `deepcoder_1_5b_ngu05_n8_k16_seed2` | 8 | 16 | 0.5 | 2 | [01KYD3X5BY0BSGP2BCDTB7E24H](https://beaker.org/ex/01KYD3X5BY0BSGP2BCDTB7E24H) |
| `deepcoder_1_5b_ngu05_n8_k16_seed3` | 8 | 16 | 0.5 | 3 | [01KYD3XPKMP6HT85G6DP6AJV3N](https://beaker.org/ex/01KYD3XPKMP6HT85G6DP6AJV3N) |
| `deepcoder_1_5b_ngu075_n8_k16_seed2` | 8 | 16 | 0.75 | 2 | [01KYD3Y6R8J2JTQG9ESMJMT9VX](https://beaker.org/ex/01KYD3Y6R8J2JTQG9ESMJMT9VX) |
| `deepcoder_1_5b_ngu075_n8_k16_seed3` | 8 | 16 | 0.75 | 3 | [01KYD3YRPS9WGBTGZCC4BYJTFW](https://beaker.org/ex/01KYD3YRPS9WGBTGZCC4BYJTFW) |
| `deepcoder_1_5b_ngu0875_n8_k16_seed2` | 8 | 16 | 0.875 | 2 | [01KYD3Z9S97MKTV0FHTTXAVSPZ](https://beaker.org/ex/01KYD3Z9S97MKTV0FHTTXAVSPZ) |
| `deepcoder_1_5b_ngu0875_n8_k16_seed3` | 8 | 16 | 0.875 | 3 | [01KYD3ZS9A83RDWS9JBSDS3QA0](https://beaker.org/ex/01KYD3ZS9A83RDWS9JBSDS3QA0) |

All 12 jobs confirmed scheduled/started with no exit codes as of this
writing (no crashes at launch time); step-level progress not yet checked.

### Launch commands

```bash
export WORKSPACE=ai2/oe-adapt-code PRIORITY=high

EXP=baseline_n8_k16_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh --seed 2
EXP=baseline_n8_k16_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh --seed 3

EXP=baseline_n4_k32_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 --seed 2
EXP=baseline_n4_k32_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 4 --num_samples_per_prompt_rollout 32 --seed 3

EXP=baseline_n2_k64_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 2 --num_samples_per_prompt_rollout 64 --seed 2
EXP=baseline_n2_k64_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --num_unique_prompts_rollout 2 --num_samples_per_prompt_rollout 64 --seed 3

EXP=ngu05_n8_k16_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.5 --seed 2
EXP=ngu05_n8_k16_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.5 --seed 3

EXP=ngu075_n8_k16_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.75 --seed 2
EXP=ngu075_n8_k16_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.75 --seed 3

EXP=ngu0875_n8_k16_seed2 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.875 --seed 2
EXP=ngu0875_n8_k16_seed3 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --never_give_up 0.875 --seed 3
```

## 2026-07-25: DeepCoder-1.5B eval-only baseline (initial model, grpo.py, OC)

Single `--eval_only` run of the untrained initial model
(`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) on `deepcoder_1_5b.sh`'s held-out
eval sets (`mnoukhov/deepcoder_lcbv5_test`, `mnoukhov/deepcoder_codeforces_test`),
to get a pre-training baseline score to compare the K/NGU sweep runs against.
`--eval_temperature 0.6` is already the script's default (set in the commit
logged above), so no override needed. `--eval_only` skips learner/FSDP setup
entirely (`grpo.py` only builds vLLM engines in this mode), so `NUM_GPUS` was
reduced to 4 (matching `--vllm_num_engines 4`) instead of the training
default of 8; `--send_slack_alerts False` to keep it quiet for a one-off
check (learner-only flags like `--fsdp_shard_degree`/`--num_learners_per_node`
are baked into the script but unused/harmless in eval-only mode).

```
NUM_GPUS=4 EXP=eval_only_initial ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --eval_only \
  --send_slack_alerts False
```

Beaker: [01KYDC2HZ0TV4TNHJ9RKWMGRMC](https://beaker.org/ex/01KYDC2HZ0TV4TNHJ9RKWMGRMC)
(exitCode=0, ~10 min runtime). wandb: [q9gelapy](https://wandb.ai/ai2-llm/open_instruct_internal/runs/q9gelapy).

**Result:** `eval scores: 1.13`, avg sequence length 13565 tokens, at
`eval_step=1` (labels the initial/pre-training model).

### Sweep-wide crash investigation: three separate root causes found and fixed

All 12 seed2/3 jobs above died (exit 1, one also preempted) within hours of
launch, and the same signature also killed one of the original 6 seed-1 jobs
(`ngu075_n8_k16`, at 77.6%) and nearly a second (`ngu0875_n8_k16`, still
running at time of writing). Two genuinely separate infra bugs were involved.

**Root cause 1 (fixed, commit `1ae99daa6`): shared AWS code-execution API
overloaded by sweep concurrency.** `deepcoder_1_5b.sh` hardcoded
`--code_api_url` to a shared AWS API Gateway/Lambda endpoint
(`p9f1719l7f.execute-api.us-west-2.amazonaws.com`) also used by other
sweeps/users. Once concurrency hit 18 jobs (6 original + 12 new seeds), the
endpoint started returning `500 Internal Server Error` on code-verification
calls (`ground_truth_utils.py:980`), stalling a rank long enough to trip a
30-minute Gloo collective timeout (`RuntimeError: Application timeout caused
pair closure`) and kill training. Every one of the 11 non-preempted crashes
showed this exact `500 Server Error` immediately preceding the timeout.
Fixed by switching `deepcoder_1_5b.sh` to the per-job local code-server
pattern already used by other code-RL scripts (`grpo_fast_7b_code.sh`, the
`olmo3` RL scripts): source `configs/beaker_configs/code_api_setup.sh`
(spins up a local uvicorn/nginx code server per job) instead of hitting the
shared endpoint, and point `--code_api_url` at `$CODE_API_URL/test_program`.

All 12 seed2/3 jobs plus `ngu075_n8_k16` seed1 were relaunched resumed from
their last saved checkpoint (`--checkpoint_state_dir`, `mason.py` skips its
`--auto_checkpoint_state_dir` override when the flag is already set to a
`/weka/`-prefixed path) with this fix, on `ai2/oe-adapt-code`/`high`. Two of
the 13 didn't get scheduled on `ai2/jupiter` (queue pressure); canceled and
relaunched with `CLUSTER="ai2/jupiter ai2/ceres"` (`mason.py --cluster`
accepts multiple values via `nargs="+"`), both scheduled immediately after.

| Job | Beaker |
| --- | --- |
| `baseline_n8_k16_seed2_resume` | [01KYEKH7VYMH71H5HEB4W46K80](https://beaker.org/ex/01KYEKH7VYMH71H5HEB4W46K80) |
| `baseline_n8_k16_seed3_resume` | [01KYEKHQV8VRV7BCPECG6FYXPK](https://beaker.org/ex/01KYEKHQV8VRV7BCPECG6FYXPK) |
| `baseline_n4_k32_seed2_resume` | [01KYEKJ6HQS6KAKV1X54QH60TC](https://beaker.org/ex/01KYEKJ6HQS6KAKV1X54QH60TC) |
| `baseline_n4_k32_seed3_resume` | [01KYEKJNFD0R4BS3BTH44J1BXF](https://beaker.org/ex/01KYEKJNFD0R4BS3BTH44J1BXF) |
| `baseline_n2_k64_seed2_resume` | [01KYEKK4FTZ466FTY72A1TWM0A](https://beaker.org/ex/01KYEKK4FTZ466FTY72A1TWM0A) |
| `baseline_n2_k64_seed3_resume` | [01KYEKKM0QABE5KNVNK5D7VZGE](https://beaker.org/ex/01KYEKKM0QABE5KNVNK5D7VZGE) |
| `ngu05_n8_k16_seed2_resume` | [01KYEKM2F1TB6ZHNPKEZWN4MWX](https://beaker.org/ex/01KYEKM2F1TB6ZHNPKEZWN4MWX) |
| `ngu05_n8_k16_seed3_resume` | [01KYEKMHTQD05NHRS3D6PAXGEC](https://beaker.org/ex/01KYEKMHTQD05NHRS3D6PAXGEC) |
| `ngu075_n8_k16_seed2_resume` | [01KYEKN27EQR12C2PPSCCBWDJY](https://beaker.org/ex/01KYEKN27EQR12C2PPSCCBWDJY) |
| `ngu075_n8_k16_seed3_resume` | [01KYEKNJ0JFY4TRWFC63PKHJYE](https://beaker.org/ex/01KYEKNJ0JFY4TRWFC63PKHJYE) |
| `ngu0875_n8_k16_seed2_resume` | [01KYEKP2MS835XQG5KT53K86GP](https://beaker.org/ex/01KYEKP2MS835XQG5KT53K86GP) |
| `ngu0875_n8_k16_seed3_resume2` (jupiter+ceres) | [01KYEM59SPZZBS4VXX1P4PD6N7](https://beaker.org/ex/01KYEM59SPZZBS4VXX1P4PD6N7) |
| `ngu075_n8_k16_seed1_resume2` (jupiter+ceres) | [01KYEM5RX7SAMQ7E30BER2AGGX](https://beaker.org/ex/01KYEM5RX7SAMQ7E30BER2AGGX) |

**Root cause 2 (fixed, commit `020a93ee0`): olmo-core's bookkeeping process
group silently ignores `--backend_timeout`.** All 13 resumed jobs above died
again anyway, spread over an 8-hour window (07:27-15:16), same
`Application timeout caused pair closure` signature but now with zero `500
Server Error` lines anywhere (confirming root cause 1's fix held). Traced
the crashing collective (`olmo_core/train/utils.py:320`, called from
`Trainer._reduce_and_pass_on_metrics`) to a *separate* CPU/gloo process group
olmo-core creates for async metric bookkeeping via bare
`torch.distributed.new_group()` (`trainer.py:375`) — no `timeout` argument.
Per `torch/distributed/distributed_c10d.py`'s `_get_default_timeout`, a
`new_group()` call with no explicit timeout resolves to the hardcoded
module-level `default_pg_timeout` constant (`0:30:00`) for any non-NCCL
backend, completely independent of whatever timeout was passed to the
original `init_process_group`/`prepare_training_environment` call. So
`--backend_timeout` (already 120 min by default) was only ever protecting
the main world process group — the bookkeeping subgroup where these crashes
actually originate was always on torch's hardcoded 30-minute default,
regardless of what we configured. Confirmed by grep: no code anywhere in
open_instruct or olmo-core passes a timeout to this specific `new_group()`
call.

Fixed in `grpo_olmo_core_actor.py`'s `setup()`: right after
`train.prepare_training_environment(...)`, overwrite
`torch.distributed.distributed_c10d.default_pg_timeout` to
`timedelta(minutes=self.grpo_config.backend_timeout)`. Any subgroup created
afterward without an explicit timeout (including olmo-core's bookkeeping
group) now inherits the same generous timeout as the main group, via the
existing `--backend_timeout` flag rather than adding a new one. Verified the
attribute is reachable and mutable post-import
(`torch.distributed.distributed_c10d.default_pg_timeout`) in the project
venv; `make style && make quality` clean. No offline-testable surface (Ray
actor distributed setup, GPU-only) — verification is via the next relaunch's
survival time.

**Monitoring note:** built `watch_sweep.sh`, a poller that checks all active
jobs' Beaker exit codes every 4 min and only emits on a state transition to
non-zero. First version had a bug — it treated the *first* observation of
any job's exit code as a transition, so a job that had already finished
successfully (`ngu0875_n8_k16` seed1, exit 0) got misreported as a crash.
Fixed to record a silent baseline on first poll and only alert on genuine
transitions (and never on exit 0).

**Relaunch attempt with the `--backend_timeout` fix (commit `020a93ee0`):**
launched all 13 resumed from checkpoint on `ai2/jupiter ai2/ceres`/`high`;
9 of 13 never got scheduled (queue pressure) and were canceled + relaunched
on `ai2/olmo-instruct`/`urgent` instead — all 13 scheduled within seconds
after that switch. One (`baseline_n4_k32_seed3`) got extremely close before
dying again: `[99.4% complete (step 497/500)]`, same
`Application timeout caused pair closure` signature, ~4h4m survival (up from
the ~3h pre-fix average — consistent with the timeout fix buying real
headroom, just not infinite).

**Root cause 3 (fixed, commit `d703db162`): nginx proxy timeout silently
truncating multi-test verification, corrupting reward signal (not just
crashing jobs).** While investigating that near-miss crash, its log also
showed `504 Server Error: Gateway Time-out for url:
http://127.0.0.1:8070/test_program_stdio` right before the Gloo timeout.
Traced to `code_utils.py`'s `get_successful_tests_stdio`: tests for one
problem run *sequentially* in a single subprocess, and the function
deliberately sizes its own internal timeout as
`max_execution_time * test_ct + 5.0` — for a 15-test problem at the default
`code_max_execution_time=1.0s`, that's ~20s, by design. But
`code_api_setup.sh`'s nginx load balancer hardcoded `proxy_read_timeout 5s`
(and `proxy_send_timeout 5s`) — nginx kills the connection and returns 504
long before a legitimate multi-test request can finish, silently zeroing
reward for otherwise-correct solutions (caught by
`ground_truth_utils.py`'s broad `except Exception` → `score=0.0`, so it
never surfaces as a visible error, just quietly worse training signal).
This is a shared setup script also used by `grpo_fast_7b_code.sh` and the
`olmo3` RL scripts, so the same silent truncation likely affected those too.
Fixed by raising both to `300s`, matching the client's own `http_timeout`
ceiling (`ground_truth_utils.py`) so nginx is never the binding constraint.

Relaunched the one crashed job (`baseline_n4_k32_seed3`, resumed from its
`step400` checkpoint) with all three fixes now in place: first attempt
([01KYH6KZR9HNNQH5SB8N1WEQC8](https://beaker.org/ex/01KYH6KZR9HNNQH5SB8N1WEQC8))
didn't get scheduled on `ai2/jupiter ai2/ceres`/`high`, canceled and
relaunched on `ai2/olmo-instruct`/`urgent`
([01KYH77B8MNFXWZGBPQ3ZWGNPJ](https://beaker.org/ex/01KYH77B8MNFXWZGBPQ3ZWGNPJ)),
scheduled immediately.

Final set of 13 active jobs (all three fixes applied):

| Job | Beaker |
| --- | --- |
| `baseline_n8_k16_seed2_resume3` | [01KYGR4AJTG2BV7R81S3F7EDP3](https://beaker.org/ex/01KYGR4AJTG2BV7R81S3F7EDP3) |
| `baseline_n8_k16_seed3_resume3` | [01KYGR4SFGZ79H7PFXJEM4E6RB](https://beaker.org/ex/01KYGR4SFGZ79H7PFXJEM4E6RB) |
| `baseline_n4_k32_seed2_resume3` | [01KYGR58F9GC91PRAP4HPKEZ41](https://beaker.org/ex/01KYGR58F9GC91PRAP4HPKEZ41) |
| `baseline_n4_k32_seed3_resume4urgent` | [01KYH77B8MNFXWZGBPQ3ZWGNPJ](https://beaker.org/ex/01KYH77B8MNFXWZGBPQ3ZWGNPJ) |
| `baseline_n2_k64_seed2_resume3urgent` | [01KYGRT9R2842E4YYCNGHSX988](https://beaker.org/ex/01KYGRT9R2842E4YYCNGHSX988) |
| `baseline_n2_k64_seed3_resume3urgent` | [01KYGRTS6HS07XT0H0Q9T9B3NK](https://beaker.org/ex/01KYGRTS6HS07XT0H0Q9T9B3NK) |
| `ngu05_n8_k16_seed2_resume3urgent` | [01KYGRV9CX4FQ68FS31SEHWF31](https://beaker.org/ex/01KYGRV9CX4FQ68FS31SEHWF31) |
| `ngu05_n8_k16_seed3_resume3urgent` | [01KYGRVRJ69EP7PRJT9GYSFMM6](https://beaker.org/ex/01KYGRVRJ69EP7PRJT9GYSFMM6) |
| `ngu075_n8_k16_seed2_resume3urgent` | [01KYGRW6XBBKXVVZH2TJ1JR0HW](https://beaker.org/ex/01KYGRW6XBBKXVVZH2TJ1JR0HW) |
| `ngu075_n8_k16_seed3_resume3urgent` | [01KYGRWNGKSZ0M452X2JG6YPSV](https://beaker.org/ex/01KYGRWNGKSZ0M452X2JG6YPSV) |
| `ngu0875_n8_k16_seed2_resume3urgent` | [01KYGRX4FGET8RENTKQC0GFR1F](https://beaker.org/ex/01KYGRX4FGET8RENTKQC0GFR1F) |
| `ngu0875_n8_k16_seed3_resume3urgent` | [01KYGRXKVYHM4HT5W6THM8XGGM](https://beaker.org/ex/01KYGRXKVYHM4HT5W6THM8XGGM) |
| `ngu075_n8_k16_seed1_resume3urgent` | [01KYGRY3DTG585YQ00JQKFHHCP](https://beaker.org/ex/01KYGRY3DTG585YQ00JQKFHHCP) |

**Next:** relaunch resumed from the 13 checkpoints again with commit
`020a93ee0`'s fix, and watch survival time past the ~3-8h window that killed
the previous two attempts.

**Ongoing crash/resume cycle (ad-hoc, via `watch_sweep.sh`):** the sweep has
continued hitting the same `Application timeout caused pair closure`
signature sporadically since the three fixes landed — survival times have
grown (several hours to ~7-8h) but crashes haven't stopped entirely, so each
one gets resumed from its last checkpoint and the watcher's job list gets
updated in place. Individual resume rounds between this point and
2026-07-27 (arms now at resume5 through resume9 depending on how many times
each has crashed) weren't all logged here individually. Latest instance:

- `ngu0875_n8_k16_seed2_resume8` ([01KYHKF079FT7K3QDN1JEA02HP](https://beaker.org/ex/01KYHKF079FT7K3QDN1JEA02HP))
  crashed 2026-07-27 18:36 at step 338/500 (67.6%), same signature, ~7h40m
  survival (started 10:56). Relaunched as
  `ngu0875_n8_k16_seed2_resume9` ([01KYJE89VVESNMDM1Y5Y43ABVY](https://beaker.org/ex/01KYJE89VVESNMDM1Y5Y43ABVY))
  from the same `--checkpoint_state_dir`, `ai2/jupiter ai2/ceres`/`urgent`,
  scheduled immediately.

Current 10-job watch list (`watch_sweep.sh`, polls every 4 min): `baseline_n8_k16_seed2_resume3`,
`baseline_n8_k16_seed3_resume3`, `baseline_n4_k32_seed2_resume3`,
`baseline_n4_k32_seed3_resume4urgent`, `ngu05_n8_k16_seed2_resume5`,
`ngu05_n8_k16_seed3_resume5`, `ngu075_n8_k16_seed2_resume7`,
`ngu075_n8_k16_seed3_resume6`, `ngu0875_n8_k16_seed2_resume9`,
`ngu0875_n8_k16_seed3_resume9`.

## 2026-07-27: Separate per-source eval metrics + HumanEvalPlus eval set (code RLVR)

### Problem

`eval/objective/code_stdio_correct_rate` (and the eval pass@k breakdown) was pooling
`lcbv5_test` and `codeforces_test` into one number instead of reporting them separately,
even though `EVAL_DATASETS` in `deepcoder_1_5b.sh` lists them as two datasets.

Root cause: `scripts/data/create_deepcoder_data.py`'s `to_example()` hardcoded
`"dataset": "code_stdio"` for every row across all four DeepCoder configs (lcbv5,
primeintellect, taco, codeforces) -- train and eval alike. That `"dataset"` column is
`VERIFIER_SOURCE_KEY` (`dataset_transformation.py`), used for two things at once:
reward-function routing (`resolve_reward_function` in `ground_truth_utils.py`) and the
per-dataset metric grouping key (`dataset_metric_key` in `grpo_utils.py`/`data_loader.py`).
Since both eval sets carried the identical literal `"code_stdio"`, every metric that groups
by that field collapsed them into one bucket -- `grpo_utils.py`'s
`if len(unique_dataset_keys) > 1:` breakout branch never fired.

Math datasets don't have this problem only because different math eval sources (e.g.
`gsm8k` vs `math`) already happen to carry different `VERIFIER_SOURCE_KEY` strings --
there's no separation *mechanism* being exercised there beyond that.

### Fix

1. `ground_truth_utils.py::resolve_reward_function`: added a prefix fallback (mirroring
   the existing `math`/`gsm8k` fallback) so any `"code_stdio*"`-prefixed name routes to the
   `code_stdio` verifier, and any other `"code*"`-prefixed name (checked after `code_stdio`,
   since `"code_stdio_lcbv5".startswith("code")` is also true) routes to the base `code`
   verifier.
2. `create_deepcoder_data.py`: `to_example()` now takes a `dataset_tag` param. Train splits
   keep `"code_stdio"` (unchanged, so training reward curves stay pooled). The held-out eval
   splits get distinct tags: `lcbv5_test` -> `"code_stdio_lcbv5"`, `codeforces_test` ->
   `"code_stdio_codeforces"`.
3. New `scripts/data/create_humanevalplus_data.py`: converts `evalplus/humanevalplus`
   (164 problems) to open-instruct RLVR format. Function-signature style (not
   stdin/stdout), so it's graded by the base `code` verifier (`POST /test_program`,
   assert-style exec) rather than `code_stdio`. Each example's ground truth is a
   single combined test string: `sample["test"] + f"\ncheck({sample['entry_point']})\n"`
   (the full evalplus check harness, including the "plus" extra tests) -- binary pass/fail
   per problem, matching standard HumanEval(+) pass@1 semantics. Tagged
   `"dataset": "code_humanevalplus"`. Pushed as `mnoukhov/humanevalplus_test`.
4. `deepcoder_1_5b.sh`: added `mnoukhov/humanevalplus_test 1.0` to `EVAL_DATASETS`.
5. Added regression tests in `test_ground_truth_utils.py` for both new prefix routes,
   including one that specifically checks `code_humanevalplus` resolves to `code` and not
   `code_stdio` (ordering matters in the fallback chain).

### Verification

- Ran the new humanevalplus conversion against the real code-exec sandbox
  (`get_successful_tests_fast`): a correct canonical solution scores 1, a wrong one scores 0.
  Spot-checked 15 random canonical solutions (all pass) and then the full 164: 162/164 pass
  at a generous timeout.
- Two "failures" at generous timeout, investigated individually:
  - `HumanEval/32` (`find_zero`): a genuine bug in evalplus's own "plus" test harness --
    `assert _poly(*candidate(*inp), inp) <= 0.0001` tries to unpack a scalar float return
    with `*`. Pre-existing upstream data issue, not something our conversion introduced.
  - `HumanEval/139`: not a bug, just slow (~10.3s wall time for the combined
    check-with-all-plus-tests call).
- **Caveat for the launch config:** `code_max_execution_time` (default 1.0s) is applied
  *per test-list entry* in `get_successful_tests_fast`, and since each HumanEvalPlus
  problem is one entry bundling all "plus" inputs, compute-heavy problems can get killed
  before finishing. At the default 1.0s budget, 8/164 canonical solutions (~5%) time out as
  false negatives (vs. 2/164 genuine issues at a generous budget). This is a systematic
  handful-of-percentage-points underestimate on the humanevalplus eval curve specifically.
  Didn't change the global default because `code_max_execution_time` is shared with
  `code_stdio`, where it scales the *total* per-problem timeout
  (`max_execution_time * test_ct + 5.0`) -- raising it to fix humanevalplus would also
  proportionally slow down every lcbv5/codeforces stdio verification during training.
  Left as a known tradeoff for whoever tunes this next; not fixed in this pass.
- Re-ran `create_deepcoder_data.py` and confirmed on the Hub: train sets (`deepcoder_lcbv5`,
  `deepcoder_primeintellect`, `deepcoder_taco`) unchanged (`{"code_stdio"}`, no-op commits);
  eval sets updated (`deepcoder_lcbv5_test` -> `{"code_stdio_lcbv5"}`,
  `deepcoder_codeforces_test` -> `{"code_stdio_codeforces"}`); new
  `humanevalplus_test` -> `{"code_humanevalplus"}`.
- `uv run pytest open_instruct/test_ground_truth_utils.py`: 54 passed.
- `make style && make quality`: clean.

### Files changed

- `open_instruct/ground_truth_utils.py` (prefix fallback)
- `open_instruct/test_ground_truth_utils.py` (2 new regression tests)
- `scripts/data/create_deepcoder_data.py` (per-source eval tags)
- `scripts/data/create_humanevalplus_data.py` (new)
- `scripts/train/qwen/deepcoder_1_5b.sh` (added humanevalplus_test to EVAL_DATASETS)

## 2026-07-27: Eval-only checkpoint comparison — 3 baselines + 3 NGU (n8_k16), lcbv5 + humanevalplus

`--eval_only` runs (4 GPUs each, `WORKSPACE=ai2/open-instruct-dev`, `PRIORITY=urgent`,
`CLUSTER` left at script default `ai2/jupiter`) against the actual saved HF-format
checkpoints from 6 of the K/NGU sweep runs, restricted to
`--dataset_mixer_eval_list mnoukhov/deepcoder_lcbv5_test 1.0 mnoukhov/humanevalplus_test 1.0`
(per user request, to see these two report separately post-fix — see the section above).

Checkpoint selection: "just the last one" per run, since only 3/6 runs have actually reached
the full 500-step target so far (the other 3 are still mid-sweep, cycling through the ongoing
crash/resume loop tracked by `watch_sweep.sh`). Checkpoints live under
`/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/` (mason's
`--auto_output_dir_path` default); found by walking `*_checkpoints/step_N` dirs across all
resume-tagged reruns of each config and taking the max N, since `setup_runtime_variables`
mints a fresh timestamped `run_name` (hence a fresh checkpoint dir) on every relaunch and step
count only continues monotonically *across* those dirs via checkpoint-state resume.

| Config | Seed | Checkpoint | Step | Beaker |
|---|---|---|---|---|
| baseline_n8_k16 | 1 | `..._baseline_n8_k16__1__1784967249` (final save; Beaker job succeeded) | 500 (complete) | [01KYJTJ9EKDR8TY4R3M25SDY4V](https://beaker.org/ex/01KYJTJ9EKDR8TY4R3M25SDY4V) |
| baseline_n8_k16 | 2 | `..._baseline_n8_k16_seed2_resume3__2__1785154849_checkpoints/step_500` | 500 (complete) | [01KYJTK89GQH71M7ED63SP9MZ1](https://beaker.org/ex/01KYJTK89GQH71M7ED63SP9MZ1) |
| baseline_n8_k16 | 3 | `..._baseline_n8_k16_seed3_resume3__3__1785158507_checkpoints/step_300` | 300 (mid-sweep, latest available) | [01KYJTM58ZPFTVDK6C9WF082CX](https://beaker.org/ex/01KYJTM58ZPFTVDK6C9WF082CX) |
| ngu05_n8_k16 | 1 | `..._ngu05_n8_k16__1__1784967885` (final save; Beaker job succeeded) | 500 (complete) | [01KYJTMX0Z8KWXYYD531679954](https://beaker.org/ex/01KYJTMX0Z8KWXYYD531679954) |
| ngu075_n8_k16 | 1 | `..._ngu075_n8_k16_seed1_resume3urgent__1__1785122081_checkpoints/step_500` | 500 (complete, after 2 crash-resumes) | [01KYJTNN43N39B4Q2QJ36EZ455](https://beaker.org/ex/01KYJTNN43N39B4Q2QJ36EZ455) |
| ngu0875_n8_k16 | 1 | `..._ngu0875_n8_k16__1__1784968037` (final save; Beaker job succeeded) | 500 (complete) | [01KYJTPBM7E74H4AX04NGT7YQZ](https://beaker.org/ex/01KYJTPBM7E74H4AX04NGT7YQZ) |

Note: "three baselines and three NGUs" was read as the 3 baseline seeds (1/2/3) + the 3 NGU
p-values (0.5/0.75/0.875) at seed1 -- seed1 happens to be where all 4 arms (baseline + 3 NGU)
already completed the full 500-step run, making it the natural "one representative per NGU
value" choice. Seed2/seed3 for the NGU arms were *not* run here; only baseline got all 3 seeds
evaluated. Flag if a different 6-way split was intended.

### Launch commands

```
export WORKSPACE=ai2/open-instruct-dev PRIORITY=urgent NUM_GPUS=4

EXP=eval_baseline_n8_k16_seed1 ./scripts/train/build_image_and_launch.sh scripts/train/qwen/deepcoder_1_5b.sh \
  --eval_only --model_name_or_path <checkpoint path> \
  --dataset_mixer_eval_list mnoukhov/deepcoder_lcbv5_test 1.0 mnoukhov/humanevalplus_test 1.0 \
  --send_slack_alerts False
# ...repeated per checkpoint above (EXP=eval_baseline_n8_k16_seed2/seed3, eval_ngu05/075/0875_n8_k16_seed1)
```

All 6 confirmed `pending` (queued on jupiter) immediately after launch, no arg-parse errors.

## 2026-07-27: Eval-only checkpoint comparison results — 3 baselines + 3 NGU (n8_k16)

All 6 jobs from the section above ran to completion (`exitCode=0` for all). Jupiter queue time
was long (~37 min pending before the first job started; last job finished ~64 min after launch),
but each eval itself took only ~2-13 min once scheduled, consistent with the ~10 min/run estimate.
Metrics pulled directly from each job's stdout rich-metrics table (`model_utils.print_rich_single_line_metrics`,
logged once per job right after `📊 Evaluation responses received`) via `beaker experiment logs <id>`
— wandb links included below for anyone who wants full curves/samples but weren't otherwise queried.

| Config | Seed | Checkpoint step | eval/scores | lcbv5 pass@1 | humanevalplus pass@1 | Beaker | wandb | Status |
|---|---|---|---|---|---|---|---|---|
| baseline_n8_k16 | 1 | 500 (complete) | 1.44 | 0.21 | 0.00 | [01KYJTJ9EKDR8TY4R3M25SDY4V](https://beaker.org/ex/01KYJTJ9EKDR8TY4R3M25SDY4V) | [r07owo48](https://wandb.ai/ai2-llm/open_instruct_internal/runs/r07owo48) | succeeded |
| baseline_n8_k16 | 2 | 500 (complete) | 1.50 | 0.19 | 0.00 | [01KYJTK89GQH71M7ED63SP9MZ1](https://beaker.org/ex/01KYJTK89GQH71M7ED63SP9MZ1) | [w1c4m3gq](https://wandb.ai/ai2-llm/open_instruct_internal/runs/w1c4m3gq) | succeeded |
| baseline_n8_k16 | 3 | 300 (mid-sweep, partial) | 1.40 | 0.18 | 0.00 | [01KYJTM58ZPFTVDK6C9WF082CX](https://beaker.org/ex/01KYJTM58ZPFTVDK6C9WF082CX) | [syvzps7d](https://wandb.ai/ai2-llm/open_instruct_internal/runs/syvzps7d) | succeeded |
| ngu05_n8_k16 | 1 | 500 (complete) | 1.25 | 0.15 | 0.00 | [01KYJTMX0Z8KWXYYD531679954](https://beaker.org/ex/01KYJTMX0Z8KWXYYD531679954) | [sdm5xbz5](https://wandb.ai/ai2-llm/open_instruct_internal/runs/sdm5xbz5) | succeeded |
| ngu075_n8_k16 | 1 | 500 (complete) | 1.38 | 0.17 | 0.00 | [01KYJTNN43N39B4Q2QJ36EZ455](https://beaker.org/ex/01KYJTNN43N39B4Q2QJ36EZ455) | [jb9c2msw](https://wandb.ai/ai2-llm/open_instruct_internal/runs/jb9c2msw) | succeeded |
| ngu0875_n8_k16 | 1 | 500 (complete) | **0.00 (invalid, see below)** | **0.00 (invalid)** | **0.00 (invalid)** | [01KYJTPBM7E74H4AX04NGT7YQZ](https://beaker.org/ex/01KYJTPBM7E74H4AX04NGT7YQZ) | [k9l3eie9](https://wandb.ai/ai2-llm/open_instruct_internal/runs/k9l3eie9) | succeeded (job), eval data corrupted |

(`lcbv5`/`humanevalplus` columns are `eval/pass_at_1/code_stdio_lcbv5` and
`eval/pass_at_1/code_humanevalplus` respectively — the per-source eval tags added earlier the
same day. `pass_at_1_unbiased` matched `pass_at_1` exactly in every run, as expected at
`eval_pass_at_k=1`.)

### `ngu0875_n8_k16_seed1` result is invalid — local code-exec API died mid-eval

Every metric for this one job is a flat `0.00e+00` (`eval/scores`, `objective/verifiable_reward`,
`objective/code_reward`, `objective/code_stdio_reward`, `pass_at_1`, both per-dataset pass@1s —
all zero), while `stop_rate=0.99` and `sequence_lengths` (mean 8131, matching the other 5 runs)
show the model generated normal-looking completions. Root cause, from the job's stdout: the
per-job local nginx code-execution load balancer (`code_api_setup.sh`, port 8070) started and
passed its health check fine at launch (`23:24:14 ✓ CODE_API_URL is responding correctly`), but
by `23:26:46` every `/test_program` and `/test_program_stdio` call started failing with
`ConnectionResetError` then `Connection refused` (`Failed to establish a new connection: [Errno
111]`), and **never recovered** for the rest of the run (115 consecutive connection-error log
lines through to the final eval at `23:29:14`). None of the other 5 jobs show a single
connection-error line in their logs, so this looks like an isolated silent nginx/local-code-server
crash specific to this one job, not a systemic regression of the `code_api_setup.sh` fix from
`1ae99daa6`. Every verification request during the outage window was scored `0.0` by
`ground_truth_utils`' broad exception handling — the same "silent reward corruption on
verifier-infra failure" failure mode already flagged in the sweep-wide crash investigation
(see [that section](#sweep-wide-crash-investigation-three-separate-root-causes-found-and-fixed)),
just triggered by a different underlying infra fault (nginx dying, not the 5s `proxy_read_timeout`
or the shared-API `500`s). This job's Beaker exit code is 0 (no crash, ran to completion), so
nothing here would have surfaced without checking the actual eval numbers.
**`ngu0875_n8_k16` needs a re-run before drawing any conclusion about `p=0.875`.**

### Summary across the 5 valid runs

`baseline_n8_k16` has the best `lcbv5` pass@1 and best aggregate `eval/scores` at every seed
checked (seed1: 0.21 / 1.44, seed2: 0.19 / 1.50, seed3 @ step 300: 0.18 / 1.40), beating both
`ngu05_n8_k16` (0.15 / 1.25) and `ngu075_n8_k16` (0.17 / 1.38) at seed1 — i.e. in this
single-seed-per-NGU-arm comparison, NGU regularization looks like it costs lcbv5 performance
rather than helping, the opposite of the deepscaler finding. This is not a controlled comparison
(only baseline has 3 seeds; NGU arms have 1 each, and baseline seed3 is a partial 300-step
checkpoint) so treat as directional only pending more NGU seeds.

`humanevalplus` pass@1 is **exactly 0.00 for all 5 valid runs**, not just low — every one of the
164 humanevalplus problems failed verification in every run. Plausible explanation: all three
training datasets (`deepcoder_lcbv5`, `deepcoder_primeintellect`, `deepcoder_taco`) are
`code_stdio`-format (stdin/stdout competitive-programming style), so none of these checkpoints
were ever trained to produce plain function-signature-style solutions matching HumanEvalPlus's
assert-based harness — likely an output-format mismatch rather than a verifier bug (the
`code_reward`/`code_correct_rate` objective metrics, which route through the separate `code`
non-stdio verifier, are also flat zero, and the same verifier confirmed 162/164 canonical
solutions pass with only 2 real failures per the earlier humanevalplus-data verification pass, so
the harness itself works). Worth spot-checking a couple of raw completions before treating this as
fully conclusive, but there's no sign of a verifier-side bug specific to today's run.

## 2026-07-28: Full checkpoint-trajectory eval — lcbv5 pass@1 vs step, all 3 seeds × 4 configs

**Superseded — see the
["corrected, all gaps backfilled"](#2026-07-28-full-checkpoint-trajectory-eval-corrected--all-gaps-backfilled)
section below.** This section's job roster and "best step" table were written before the
`baseline_seed2`, `ngu075_seed1`, and `ngu0875_seed2` backfill jobs below had been checked; two of
those backfills turned out to hit the *same* nginx code-exec failure as the checkpoints they were
meant to replace. Job IDs, raw per-checkpoint numbers, and the failure-mode analysis below are
still accurate and are carried forward as-is; only the "what's still missing" bookkeeping and the
final ranking conclusion are superseded.

Follow-up to the single-checkpoint comparison above: instead of "just the last checkpoint,"
`--eval_only`-looped over *every* saved checkpoint (steps 100/200/.../500 as available) for all
12 (config, seed) lineages, one Beaker job per lineage, each internally sequencing through its
checkpoints via `--eval_only_set_checkpoint <step>` and logging each checkpoint as a separate
wandb run sharing a common `wandb_group_name = eval_<lineage>`. `--eval_only_set_checkpoint`
confirmed correct in the canary job (`baseline_seed1`, `eval_step: 500` matched the requested
checkpoint).

### Job roster, launches, and mid-run corrections

12 jobs were launched initially (`WORKSPACE=ai2/open-instruct-dev`, `PRIORITY=urgent`,
`CLUSTER=ai2/jupiter`). Partway through, 4 of them (`baseline_seed2`, `ngu05_seed1`,
`ngu075_seed1`, `ngu0875_seed2`) were stopped and relaunched on `ai2/oe-adapt-code` /
`PRIORITY=high` / `CLUSTER="ai2/jupiter ai2/neptune"` with new job IDs (original IDs are dead,
not queried). Additionally, two jobs (`baseline_seed3`, `ngu05_seed3`) crashed partway through
their checkpoint loop on a transient
`ValueError: Free memory on device cuda:0 (6.47/79.19 GiB) ... less than desired GPU memory
utilization` (vLLM engine re-init OOM-checking against stale allocator state from the *previous*
checkpoint's engine in the same process — see "GPU-memory crash pattern" note below) and had
their missing final checkpoint backfilled by a follow-up single-checkpoint job. Final roster:

| Lineage | Beaker job(s) | Checkpoints requested | Checkpoints obtained |
|---|---|---|---|
| baseline_seed1 | [01KYK9062B1AZFQJ2Y17SCQC13](https://beaker.org/ex/01KYK9062B1AZFQJ2Y17SCQC13) | 500 | 500 |
| baseline_seed2 | [01KYKAD77YTNCDCW0NWADDN487](https://beaker.org/ex/01KYKAD77YTNCDCW0NWADDN487) (relaunch) | 100,200,300,400,500 | 100,200 only — job failed (GPU-memory crash) attempting step 300 |
| baseline_seed3 | [01KYK98XFPE19AJRA6GTBHDX7P](https://beaker.org/ex/01KYK98XFPE19AJRA6GTBHDX7P) + step-400 backfill [01KYKAJM33VKF334JXTQCWQVZX](https://beaker.org/ex/01KYKAJM33VKF334JXTQCWQVZX) | 100,200,300,400 | all 4 (100/200/300 from first job, which failed attempting step 400; 400 from backfill job) |
| ngu05_seed1 | [01KYKADS8ZY5A5P5P3F5BHYS2C](https://beaker.org/ex/01KYKADS8ZY5A5P5P3F5BHYS2C) (relaunch) | 500 | 500 |
| ngu05_seed2 | [01KYK97Y6273EZB83X8DAN6X52](https://beaker.org/ex/01KYK97Y6273EZB83X8DAN6X52) | 100,200,300,400,500 | all 5 |
| ngu05_seed3 | [01KYK99D2TK3ZV1A4WA69JC42D](https://beaker.org/ex/01KYK99D2TK3ZV1A4WA69JC42D) + step-400 backfill [01KYKAK52BXWQTRGS9QJKD1ADV](https://beaker.org/ex/01KYKAK52BXWQTRGS9QJKD1ADV) | 100,200,300,400,500 | 100/200/300 valid; 400 obtained but **invalid** (nginx code-exec API died mid-eval, see below); 500 never attempted |
| ngu075_seed1 | [01KYKAEA22G6KWSPZ7CTBAHWFP](https://beaker.org/ex/01KYKAEA22G6KWSPZ7CTBAHWFP) (relaunch) | 400,500 | 400 valid; 500 obtained but **invalid** (same nginx failure) |
| ngu075_seed2 | [01KYK9AW65ZW7ZY81SYN1KBP5G](https://beaker.org/ex/01KYK9AW65ZW7ZY81SYN1KBP5G) | 100,200,300 | all 3 |
| ngu075_seed3 | [01KYK99WBBT67JRY1FWTGC1JV9](https://beaker.org/ex/01KYK99WBBT67JRY1FWTGC1JV9) | 100,200,300,400,500 | all 5 |
| ngu0875_seed1 | [01KYK98CZN071T79BMK37GFX3S](https://beaker.org/ex/01KYK98CZN071T79BMK37GFX3S) | 500 | 500 |
| ngu0875_seed2 | [01KYKAETTA7D1ZVNYR6MVXNHA0](https://beaker.org/ex/01KYKAETTA7D1ZVNYR6MVXNHA0) (relaunch) | 100,200,300,400 | 100 valid; 200 obtained but **invalid** (nginx failure); job then failed (GPU-memory crash) attempting step 300 — 300/400 never obtained |
| ngu0875_seed3 | [01KYK9ABX1WGN2K4G2J82BE7T8](https://beaker.org/ex/01KYK9ABX1WGN2K4G2J82BE7T8) | 100,200,300,400 | all 4 |

### Two distinct failure modes hit during this run (neither is a new root cause — both match
### previously-documented signatures)

1. **GPU-memory crash between sequential checkpoints in one job** (`baseline_seed2`,
   `baseline_seed3` original job, `ngu05_seed3` original job, `ngu0875_seed2`): all four crashes
   show the identical `vllm.../gpu_worker.py` traceback —
   `ValueError: Free memory on device cuda:0 (X/Y GiB) on startup is less than desired GPU memory
   utilization` — always on the checkpoint *after* the 2nd-3rd one loaded in that job's process
   (step 300 of a 100/200/300/... sequence, or step 400 of a 100/200/300/400 sequence). Consistent
   with each `--eval_only_set_checkpoint` pass re-initializing a vLLM engine without fully
   releasing the previous checkpoint's GPU allocation first — an apparent memory-accumulation
   issue in the multi-checkpoint eval loop, not investigated further here (no code changes made
   per task scope) but worth flagging for whoever owns `deepcoder_1_5b_eval_checkpoints_inner.sh`.
2. **Silent nginx/local-code-exec-API death → reward forced to 0.0** (`ngu05_seed3` step 400
   backfill, `ngu075_seed1` step 500, `ngu0875_seed2` step 200): same signature as the
   `ngu0875_n8_k16_seed1` invalidation from the section above —
   `ConnectionResetError`/`Failed to establish a new connection: [Errno 111] Connection refused`
   to `127.0.0.1:8070` starts partway through the job and never recovers, silently forcing every
   verification in the affected eval to `score=0.0` while the job itself still exits 0. Confirmed
   by grepping each affected log: e.g. `ngu075_seed1` shows 262 connection-error lines in the
   window between its (valid) step-400 metrics block and its (invalid) step-500 block, vs. only
   31 stray lines before step 400 (which came back with normal nonzero numbers). All three
   affected checkpoints are marked **invalid** in the table below and excluded from the "best
   step" analysis; two of them (`ngu05_seed3` step 400, `ngu0875_seed2` step 200) have no valid
   re-run in this batch.

### Full per-checkpoint results

`scores` = pooled `eval/scores`; `lcbv5` = `eval/pass_at_1/code_stdio_lcbv5`; `hep` =
`eval/pass_at_1/code_humanevalplus` (exactly 0.00 in every single checkpoint below, all 39
data points — consistent with the output-format-mismatch explanation from the section above, not
re-litigated here).

| Config | Seed | Step | scores | lcbv5 pass@1 | hep pass@1 | Note |
|---|---|---|---|---|---|---|
| baseline | 1 | 500 | 1.27 | 0.18 | 0.00 | |
| baseline | 2 | 100 | 1.25 | 0.17 | 0.00 | |
| baseline | 2 | 200 | 1.30 | 0.17 | 0.00 | job failed before step 300 |
| baseline | 3 | 100 | 1.25 | 0.18 | 0.00 | |
| baseline | 3 | 200 | 1.28 | 0.16 | 0.00 | |
| baseline | 3 | 300 | 1.34 | 0.16 | 0.00 | |
| baseline | 3 | 400 | 1.43 | 0.19 | 0.00 | from step-400 backfill job |
| ngu05 | 1 | 500 | 1.21 | 0.14 | 0.00 | |
| ngu05 | 2 | 100 | 1.35 | 0.19 | 0.00 | |
| ngu05 | 2 | 200 | 1.32 | 0.18 | 0.00 | |
| ngu05 | 2 | 300 | 1.19 | 0.15 | 0.00 | |
| ngu05 | 2 | 400 | 1.20 | 0.14 | 0.00 | |
| ngu05 | 2 | 500 | 1.31 | 0.19 | 0.00 | |
| ngu05 | 3 | 100 | 1.31 | 0.15 | 0.00 | |
| ngu05 | 3 | 200 | 1.26 | 0.15 | 0.00 | |
| ngu05 | 3 | 300 | 1.42 | 0.21 | 0.00 | |
| ngu05 | 3 | 400 | **0.00** | **0.00** | 0.00 | **INVALID** — nginx code-exec API died mid-eval |
| ngu075 | 1 | 400 | 1.27 | 0.18 | 0.00 | |
| ngu075 | 1 | 500 | **0.00** | **0.00** | 0.00 | **INVALID** — nginx code-exec API died mid-eval |
| ngu075 | 2 | 100 | 1.31 | 0.14 | 0.00 | |
| ngu075 | 2 | 200 | 1.37 | 0.18 | 0.00 | |
| ngu075 | 2 | 300 | 1.41 | 0.18 | 0.00 | |
| ngu075 | 3 | 100 | 1.30 | 0.18 | 0.00 | |
| ngu075 | 3 | 200 | 1.35 | 0.17 | 0.00 | |
| ngu075 | 3 | 300 | 1.28 | 0.15 | 0.00 | |
| ngu075 | 3 | 400 | 1.33 | 0.15 | 0.00 | |
| ngu075 | 3 | 500 | 1.30 | 0.17 | 0.00 | |
| ngu0875 | 1 | 500 | 1.33 | 0.16 | 0.00 | |
| ngu0875 | 2 | 100 | 1.24 | 0.17 | 0.00 | |
| ngu0875 | 2 | 200 | **0.00** | **0.00** | 0.00 | **INVALID** — nginx code-exec API died mid-eval; job then failed before step 300 |
| ngu0875 | 3 | 100 | 1.32 | 0.17 | 0.00 | |
| ngu0875 | 3 | 200 | 1.33 | 0.19 | 0.00 | |
| ngu0875 | 3 | 300 | 1.23 | 0.16 | 0.00 | |
| ngu0875 | 3 | 400 | 1.22 | 0.15 | 0.00 | |

### Best step per run (valid checkpoints only)

| Config | Seed | Best step (lcbv5) | Best lcbv5 pass@1 | Best step (scores) | Best scores | Valid/requested |
|---|---|---|---|---|---|---|
| baseline | 1 | 500 | 0.18 | 500 | 1.27 | 1/1 |
| baseline | 2 | 100 | 0.17 | 200 | 1.30 | 2/5 |
| baseline | 3 | 400 | **0.19** | 400 | 1.43 | 4/4 |
| ngu05 | 1 | 500 | 0.14 | 500 | 1.21 | 1/1 |
| ngu05 | 2 | 100 or 500 | 0.19 | 100 | 1.35 | 5/5 |
| ngu05 | 3 | 300 | **0.21** | 300 | 1.42 | 3/4 (400 invalid, 500 not attempted) |
| ngu075 | 1 | 400 | 0.18 | 400 | 1.27 | 1/2 (500 invalid) |
| ngu075 | 2 | 200 or 300 | 0.18 | 300 | 1.41 | 3/3 |
| ngu075 | 3 | 100 | 0.18 | 200 | 1.35 | 5/5 |
| ngu0875 | 1 | 500 | 0.16 | 500 | 1.33 | 1/1 |
| ngu0875 | 2 | 100 | 0.17 | 100 | 1.24 | 1/4 (200 invalid, 300/400 not obtained) |
| ngu0875 | 3 | 200 | **0.19** | 200 | 1.33 | 4/4 |

### Does the trajectory change NGU's ranking vs. baseline?

Not decisively, but it does complicate the single-checkpoint story from the section above.
Taking each lineage's best *valid* lcbv5 pass@1 regardless of step: baseline's ceiling in this
batch is 0.19 (seed 3, reached at step 400 of only 4 completed steps — its most-trained
checkpoint), matched by `ngu0875_seed3` (0.19 at step 200 — *half* as many steps) and beaten by
`ngu05_seed3` (0.21 at step 300). So at the level of "best checkpoint anywhere in a (possibly
truncated) trajectory," two of the three NGU values now have a seed that ties or beats every
baseline seed, reversing the previous single-checkpoint finding that baseline swept all three
NGU arms. However this is a fragile read: `ngu05_seed3` and `ngu0875_seed2`'s trajectories are
exactly the ones truncated early by infra failures (GPU-memory crash and/or invalidated nginx
checkpoints), so "best of an incomplete trajectory" is not directly comparable to "best of a
complete trajectory" — it's entirely possible baseline's own peak would also move if its seed 2/3
runs had reached step 500 uncorrupted. Within the trajectories that *did* run to completion
(`baseline_seed1`, `ngu05_seed2`, `ngu075_seed2/3`, `ngu0875_seed1/3`), the picture is noisier
than monotonic: every config's per-seed lcbv5 bounces around by ±0.03-0.05 step-to-step with no
clean upward trend by step 500 (e.g. `ngu05_seed2` peaks at step 100 *and* step 500, dips in
between; `ngu075_seed3` peaks at step 100 and never re-reaches it). No config shows a clear
late-training pull-ahead or pull-away from baseline — the single-checkpoint snapshot and the
full trajectory tell a broadly consistent "NGU roughly comparable to baseline, within run-to-run
noise" story, just with the specific ranking at any one step being fairly noisy and highly
sensitive to which checkpoints survived the infra failures above. Given how much of this batch
was truncated or invalidated, the cleanest next step is re-running the incomplete/invalidated
cells (`baseline_seed2` steps 300-500, `ngu05_seed3` steps 400-500, `ngu0875_seed2` steps
200-400) rather than drawing a firm conclusion from what's here.

## 2026-07-28: Full checkpoint-trajectory eval, corrected — all gaps backfilled

Backfill jobs for the three gaps left open by the section above have now finished:
`baseline_seed2` steps 300/400/500 ([01KYKJ2VBNG510KD6X30F7J11K](https://beaker.org/ex/01KYKJ2VBNG510KD6X30F7J11K)),
`ngu075_seed1` step 500 ([01KYKJ2VBABAMF5DQVD586X6VW](https://beaker.org/ex/01KYKJ2VBABAMF5DQVD586X6VW)),
and `ngu075_seed1` step 400 ([01KYKAEA22G6KWSPZ7CTBAHWFP](https://beaker.org/ex/01KYKAEA22G6KWSPZ7CTBAHWFP),
already valid, no reissue needed), plus `ngu0875_seed2` steps 200/300/400
([01KYKJMDYJAN92K4ZTPTV7P61S](https://beaker.org/ex/01KYKJMDYJAN92K4ZTPTV7P61S)). All were grepped
for `Connection refused` (the nginx-code-exec-death signature from the previous section) in the
window before each checkpoint's metrics block, not just checked for a completed block — see the
finding below on why a "completed" block isn't sufficient on its own.

**New finding: two of the backfill checkpoints are themselves invalid, hitting the identical
nginx code-exec failure a second time.** `ngu05_seed3` step 400's backfill
([01KYKAK52BXWQTRGS9QJKD1ADV](https://beaker.org/ex/01KYKAK52BXWQTRGS9QJKD1ADV)) and
`ngu0875_seed2` step 400's backfill (from `01KYKJMDYJAN92K4ZTPTV7P61S` above) both produced a
structurally-complete metrics block (`eval_step: 400` present, `Done step 400` printed, job exit
0) with **every** metric flat `0.00e+00` — `eval/scores`, `objective/code_stdio_reward`,
`pass_at_1`, both per-dataset pass@1s — while `sequence_lengths` still looks normal (~8300-8450,
in line with every valid run), confirming generation succeeded but verification silently died.
Both logs show a sustained run of `Connection refused` errors to `127.0.0.1:8070`
(99 lines for the `ngu05_seed3` backfill, 120 for the `ngu0875_seed2` backfill) starting well
before that checkpoint's eval and never recovering, vs. 0 such lines in every genuinely-valid job
in this batch. This means the task's original assumption — "a completed `eval_step: N` metrics
block = trustworthy data" — is not sufficient on its own; a job can complete cleanly and still log
a fully zeroed, garbage block. `ngu075_seed1` step 500's backfill and all three of
`ngu0875_seed2` steps 200/300/400 job's non-400 checkpoints (200, 300) came back clean (0
connection-refused lines each) and are trusted.

Net effect on the two lineages that needed backfills for a *missing final checkpoint*:
`ngu05_seed3` still has no valid step-400 or step-500 data (400 obtained twice now, invalid both
times; 500 never attempted) — its trajectory tops out at step 300. `ngu0875_seed2` gained valid
step 200 and step 300 data from the backfill but step 400 remains invalid — its trajectory tops
out at step 300. Every other lineage is now fully populated for its requested checkpoint range.

### Per-config trajectories (rows = seed × step)

**baseline**

| Seed | Step | eval/scores | lcbv5 pass@1 | hep pass@1 | Note |
|---|---|---|---|---|---|
| 1 | 500 | 1.27 | 0.18 | 0.00 |  |
| 2 | 100 | 1.25 | 0.17 | 0.00 |  |
| 2 | 200 | 1.30 | 0.17 | 0.00 |  |
| 2 | 300 | 1.27 | 0.15 | 0.00 |  |
| 2 | 400 | 1.34 | 0.17 | 0.00 |  |
| 2 | 500 | 1.32 | 0.16 | 0.00 |  |
| 3 | 100 | 1.25 | 0.18 | 0.00 |  |
| 3 | 200 | 1.28 | 0.16 | 0.00 |  |
| 3 | 300 | 1.34 | 0.16 | 0.00 |  |
| 3 | 400 | 1.43 | 0.19 | 0.00 |  |

**ngu05 (p=0.5)**

| Seed | Step | eval/scores | lcbv5 pass@1 | hep pass@1 | Note |
|---|---|---|---|---|---|
| 1 | 500 | 1.21 | 0.14 | 0.00 |  |
| 2 | 100 | 1.35 | 0.19 | 0.00 |  |
| 2 | 200 | 1.32 | 0.18 | 0.00 |  |
| 2 | 300 | 1.19 | 0.15 | 0.00 |  |
| 2 | 400 | 1.20 | 0.14 | 0.00 |  |
| 2 | 500 | 1.31 | 0.19 | 0.00 |  |
| 3 | 100 | 1.31 | 0.15 | 0.00 |  |
| 3 | 200 | 1.26 | 0.15 | 0.00 |  |
| 3 | 300 | 1.42 | 0.21 | 0.00 |  |
| 3 | 400 | **INVALID** | **INVALID** | **INVALID** | nginx code-exec death, backfill re-hit same failure; step 500 never attempted |

**ngu075 (p=0.75)**

| Seed | Step | eval/scores | lcbv5 pass@1 | hep pass@1 | Note |
|---|---|---|---|---|---|
| 1 | 400 | 1.27 | 0.18 | 0.00 |  |
| 1 | 500 | 1.44 | 0.19 | 0.00 |  |
| 2 | 100 | 1.31 | 0.14 | 0.00 |  |
| 2 | 200 | 1.37 | 0.18 | 0.00 |  |
| 2 | 300 | 1.41 | 0.18 | 0.00 |  |
| 3 | 100 | 1.30 | 0.18 | 0.00 |  |
| 3 | 200 | 1.35 | 0.17 | 0.00 |  |
| 3 | 300 | 1.28 | 0.15 | 0.00 |  |
| 3 | 400 | 1.33 | 0.15 | 0.00 |  |
| 3 | 500 | 1.30 | 0.17 | 0.00 |  |

**ngu0875 (p=0.875)**

| Seed | Step | eval/scores | lcbv5 pass@1 | hep pass@1 | Note |
|---|---|---|---|---|---|
| 1 | 500 | 1.33 | 0.16 | 0.00 |  |
| 2 | 100 | 1.24 | 0.17 | 0.00 |  |
| 2 | 200 | 1.33 | 0.15 | 0.00 |  |
| 2 | 300 | 1.46 | 0.19 | 0.00 |  |
| 2 | 400 | **INVALID** | **INVALID** | **INVALID** | nginx code-exec death, backfill re-hit same failure |
| 3 | 100 | 1.32 | 0.17 | 0.00 |  |
| 3 | 200 | 1.33 | 0.19 | 0.00 |  |
| 3 | 300 | 1.23 | 0.16 | 0.00 |  |
| 3 | 400 | 1.22 | 0.15 | 0.00 |  |

`hep` (`eval/pass_at_1/code_humanevalplus`) is exactly 0.00 in every one of the 37 valid
checkpoints above — same output-format-mismatch explanation as the earlier sections, not
re-litigated here.

### Best step per lineage (12 rows)

Max `pass_at_1/code_stdio_lcbv5` and max pooled `eval/scores` over all valid steps in each
lineage's trajectory (independently — the step that maximizes lcbv5 isn't always the step that
maximizes pooled scores, since pooled scores also includes the humanevalplus-zeroed and reward
magnitude terms):

| Config | Seed | Best step (lcbv5) | Best lcbv5 pass@1 | Best step (scores) | Best eval/scores |
|---|---|---|---|---|---|
| baseline | 1 | 500 | 0.18 | 500 | 1.27 |
| baseline | 2 | 100 (tie: 100/200/400) | 0.17 | 400 | 1.34 |
| baseline | 3 | 400 | **0.19** | 400 | 1.43 |
| ngu05 | 1 | 500 | 0.14 | 500 | 1.21 |
| ngu05 | 2 | 100 (tie: 100/500) | 0.19 | 100 | 1.35 |
| ngu05 | 3 | 300 | **0.21** | 300 | 1.42 |
| ngu075 | 1 | 500 | **0.19** | 500 | 1.44 |
| ngu075 | 2 | 200 (tie: 200/300) | 0.18 | 300 | 1.41 |
| ngu075 | 3 | 100 | 0.18 | 200 | 1.35 |
| ngu0875 | 1 | 500 | 0.16 | 500 | 1.33 |
| ngu0875 | 2 | 300 | **0.19** | 300 | 1.46 |
| ngu0875 | 3 | 200 | **0.19** | 200 | 1.33 |

Per-config mean of "best lcbv5 per seed": baseline 0.180, ngu05 0.180, ngu075 0.183, ngu0875
0.180 — all four configs land in the same ~0.18 band once every arm gets the same 3-seed,
best-of-trajectory treatment.

### Does NGU's ranking vs. baseline change across the trajectory, and does it change now that
### NGU has 3 seeds instead of 1?

Yes to both, and the two effects compound. Taking only step 500 (or each lineage's single
requested checkpoint, the original "just the last one" comparison from 2026-07-27) with only
seed 1 of each NGU arm — the comparison actually run on 2026-07-27 — gave baseline=0.18,
ngu05=0.14, ngu0875=0.16, both clearly below baseline, and ngu075=0.19, above it: 2 of 3 NGU
values looked worse, 1 looked better. That 2-out-of-3 pattern is exactly what drove the earlier
"NGU costs lcbv5 performance" read. With seed2/seed3 now in hand, seed 1 turns out to be
`ngu05`'s *worst* seed (0.14 vs. 0.19 and 0.21 for its other two) and `ngu0875`'s worst seed too
(0.16 vs. 0.19 and 0.19) — both by a comfortable margin. In other words, the original comparison
wasn't measuring "does NGU help or hurt," it was measuring "what does one unlucky seed look
like," and it happened to draw the low seed for 2 of the 3 NGU values. Once all three seeds are
in and each lineage is scored by its own best checkpoint (not forced to step 500, since several
lineages never reached it), every NGU value has at least one seed that ties or beats baseline's
best seed (`ngu05_seed3` 0.21 and `ngu075_seed1` 0.19 both beat baseline's best of 0.19;
`ngu0875_seed2`/`seed3` tie it at 0.19), and the four config-level means (0.180/0.180/0.183/0.180)
are indistinguishable given per-seed spread of 0.14-0.21. **Verdict: there is no reliable
evidence, at n=3 seeds/arm, that any NGU value (0.5/0.75/0.875) either helps or hurts lcbv5 pass@1
relative to baseline — seed-to-seed variance (±0.03-0.05, and up to ±0.05 for the two low-seed1
outliers) dominates any config-level effect, and the earlier single-seed "baseline wins" finding
was a seed-selection artifact, not a real NGU effect.** This doesn't rule out a real but small
effect that would need more seeds to detect, but it does mean the current data can't support
choosing an NGU value on lcbv5 grounds. Two lineages (`ngu05_seed3`, `ngu0875_seed2`) are still
capped at step 300 by the repeated nginx failure and would need a third backfill attempt to reach
full trajectories, but neither is close to being an outlier at either end, so this gap is unlikely
to change the verdict above.
