# Paper experiments

This directory is a reproduction guide for the three RLVR (RL with verifiable
rewards) experiments in the paper. Each subdirectory holds the **single training
launch script** used for that experiment, and this file explains the model, data,
and the notable non-default options each script sets.

All training runs use `open_instruct/grpo_fast.py` (async GRPO). The scripts here
were run on AI2's Beaker cluster through `mason.py`; see
[Running the scripts](#running-the-scripts) for how to adapt them to your setup.

| Experiment | Script | Base model | Train data |
|---|---|---|---|
| GSM8K | [`gsm8k/qwen2.5_0.5b_gsm8k.sh`](gsm8k/qwen2.5_0.5b_gsm8k.sh) | `Qwen/Qwen2.5-0.5B-Instruct` | `mnoukhov/gsm8k-train-harder-quartiles` |
| DeepScaleR | [`deepscaler/qwen3_4b_deepscaler.sh`](deepscaler/qwen3_4b_deepscaler.sh) | `Qwen/Qwen3-4B-Thinking-2507` | `mnoukhov/deepscaler_openinstruct` |
| Manufactoria | [`manufactoria/qwen3_4b.sh`](manufactoria/qwen3_4b.sh) | `Qwen/Qwen3-4B-Instruct-2507` | `mnoukhov/manufactoria-qwen3-4b-instruct-warmup650-pass128` |

## Setup

```bash
uv sync                      # create the environment (see the top-level README for details)
```

Runs log to Weights & Biases (`--with_tracking`). Datasets and base models are
pulled from the Hugging Face Hub.

## Running the scripts

Each script is a thin wrapper: it calls `mason.py` to schedule a Beaker job, then
runs `grpo_fast.py` inside it. The reproducible part is the `grpo_fast.py`
invocation and its flags. To run elsewhere:

- Set the Beaker placeholders via environment variables, e.g.
  `BEAKER_IMAGE=you/open_instruct WORKSPACE=... CLUSTER=... BUDGET=... bash experiments/gsm8k/qwen2.5_0.5b_gsm8k.sh`, **or**
- Drop the `mason.py ... --` prefix and run the `uv run open_instruct/grpo_fast.py ...`
  body directly under your own launcher (`accelerate`/`torchrun`), keeping the
  flags unchanged.

Extra flags are forwarded to `grpo_fast.py` (`"$@"`), so sweeps were run by
appending options on the command line.

## Shared method options

These are set (to non-default values) by all three scripts:

- `--advantage_normalization_type centered` – center grouped advantages without
  dividing by the group std.
- `--beta 0.0` (GSM8K, DeepScaleR) / `--beta 0.01` (Manufactoria) – KL penalty to
  the reference policy.
- `--async_steps 1`–`2`, `--inflight_updates` – off-policy async rollout/update
  overlap.
- `--clip_higher` – asymmetric PPO clipping upper bound (DAPO-style):
  `0.28` for GSM8K / Manufactoria, `0.272` for DeepScaleR.
- `--mask_truncated_completions False`, `--non_stop_penalty False` – keep
  length-truncated samples and do not penalize them.

The training code also adds several rollout features used in the paper's sweeps
(enable by appending the flags):

- `--never_give_up <p>` / `--never_give_up_int <n>` – requeue zero-std ("solved" or
  "unsolved") prompts as retry chains instead of dropping them, with
  `--maintain_pending_ngu_*` controlling how retried completions and baselines are
  merged.
- `--active_sampling` – oversample prompts and keep generating until a full batch
  of non-zero-std groups is collected.
- `--sync_sampling` – enqueue rollout prompts step-by-step instead of continuously
  replenishing from the dataloader.
- `--log_train_solve_rate_metrics` – log per-prompt train solve-rate metrics.

See `open_instruct/data_loader.py` (`DataLoaderConfig`) and
`open_instruct/grpo_fast.py` for the full, documented option list.

## Datasets

The pass-rate / quartile datasets are built from base pass@k rollouts. Recipes
live in [`scripts/data/rlvr/`](../scripts/data/rlvr/); the `mason.py` wrappers use
the same Beaker placeholders as above.

### GSM8K

- **Train:** `mnoukhov/gsm8k-train-harder-quartiles` (GSM8K train, bucketed by
  base-model pass rate).
- **Eval:** `mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt-quartiles`
  (GSM8K-Platinum, 1024 samples/prompt, quartile-bucketed).
- **Build:** `scripts/data/rlvr/create_pass_dataset.sh` → `gsm8k_pass_at_32_dataset.py`
  (pass@k rollouts) then `create_gsm8k_pass_rate_buckets.py` /
  `create_gsm8k_pass_rate_quartiles.py`.
- **Chat template:** `qwen_instruct_user_boxed_math`.
- Key sizes: `--response_length 4096 --max_prompt_token_length 512`,
  16 samples/prompt × 32 prompts/rollout, `--learning_rate 1e-6`, 512k episodes.

### DeepScaleR

- **Train:** `mnoukhov/deepscaler_openinstruct` – DeepScaleR-Preview converted to
  the OpenInstruct RLVR format by `scripts/data/create_deepscaler_data.py`.
- **Eval:** `mnoukhov/aime_2025_openinstruct`, `mnoukhov/brumo_2025_openinstruct`,
  `mnoukhov/hmmt_nov_2025_openinstruct`, `mnoukhov/hmmt_feb_2025_openinstruct`
  (pass@32, `--eval_top_p 0.95`).
- **Build (math eval quartiles):** `scripts/data/rlvr/aime_pass_at_k_dataset.py`
  then `create_aime_pass_rate_quartiles.py` (see also
  `create_matharena_pass_datasets.sh`).
- **Extras:** `--truncated_importance_sampling_ratio_cap 2.0`,
  `--system_prompt_override_file experiments/deepscaler/math_system_prompt.txt`,
  `--load_ref_policy True`, `--response_length 16384`, 8 samples/prompt × 32
  prompts/rollout, 256k episodes.

### Manufactoria

- **Train/eval:** `mnoukhov/manufactoria-qwen3-4b-instruct-warmup650-pass128`
  (train split for RL, first 100 test rows for eval).
- **Reward:** the Manufactoria puzzle simulator, served as an HTTP API.
  `experiments/manufactoria/manufactoria_api_setup.sh` starts a pool of
  `open_instruct.code_utils.manufactoria_api` workers behind nginx and exports
  `MANUFACTORIA_API_URL`; the script passes
  `--manufactoria_api_url $MANUFACTORIA_API_URL/test_solution --manufactoria_scoring_mode pass_rate`.
- **Build:** `scripts/data/rlvr/manufactoria_pass_at_k_dataset.py` (+
  `manufactoria_pass_at_k_qwen3_4b*.sh` wrappers, `manufactoria_pass_at_k_local.sh`
  for a single-node run). Solutions are parsed by
  `open_instruct/code_utils/manufactoria_parser.py`.
- **Extras:** 2 nodes × 8 GPUs, `--response_length 12000`, `--learning_rate 5e-7`,
  `--max_grad_norm 5`, `--num_epochs 1`, 768k episodes.
