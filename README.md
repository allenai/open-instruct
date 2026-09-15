# Learning to Solve Hard Problems in RL for LLMs by Never Giving Up

Research code for our paper **Learning to Solve Hard Problems in RL for LLMs by Never Giving Up**. In order to do better async RL, we reallocate more compute to harder prompts. Instead of discarding prompts that get all completions incorrect, the trainer requeues them for another attempt with probability `ngu`. If a later attempt finds a correct response, all completions are merged one GRPO group for training.

This is the code for reproducing the experiments in the paper. For cleaner code implementing NGU as an algorithm, check out the [PR in open-instruct](https://github.com/allenai/open-instruct/pull/1861).

See also [the paper on Arxiv](https://arxiv.org/abs/2609.13443) and [the blog post](https://mnoukhov.github.io/posts/ngu/). To cite:

```
@misc{noukhovitch_ngu_2026,
	title = {Learning to Solve Hard Problems in RL for LLMs by Never Giving Up},
	url = {https://arxiv.org/abs/2609.13443},
	author = {Noukhovitch, Michael and Ivison, Hamish and Lambert, Nathan and Courville, Aaron},
	month = sep,
	year = {2026},
}
```

## Info

This is a fork of [`allenai/open-instruct`](https://github.com/allenai/open-instruct)
(forked at commit `3d761d0`). It adds the training-loop changes, three experiment
launch scripts, and the dataset-construction scripts used in the paper.

- Upstream docs (install, Docker, `mason.py`, SFT/DPO/RLVR, evaluation) still
  apply and are preserved verbatim in **[`OPENINSTRUCT_README.md`](OPENINSTRUCT_README.md)**.
- Per-experiment models, datasets, and hyperparameters are in
  **[`experiments/README.md`](experiments/README.md)**.
- This file covers what is specific to the fork and how the never-give-up
  plumbing works.

## Repository layout

| Path | What |
|---|---|
| `experiments/gsm8k/`, `experiments/deepscaler/`, `experiments/manufactoria/` | One training launch script per paper experiment. |
| `experiments/README.md` | Reproduction guide: base model, train/eval datasets, notable flags per experiment. |
| `scripts/data/rlvr/` | Dataset construction: pass@k rollout generation and pass-rate quartile / bucket datasets, plus their Beaker wrappers. |
| `open_instruct/grpo_fast.py` | Async GRPO training entrypoint (learners + vLLM inference actors + data-prep actor + weight-sync thread). |
| `open_instruct/data_loader.py` | `StreamingDataLoader`, `DataPreparationActor`, `accumulate_inference_batches` — async rollout collection, zero-std filtering, never-give-up retries, active/sync sampling. |
| `open_instruct/data_loader_utils.py` | `compute_grouped_advantages` — centered normalization and never-give-up baseline-count rescaling. |
| `open_instruct/code_utils/manufactoria_api.py`, `manufactoria_parser.py` | Manufactoria puzzle simulator served as an HTTP reward API, and the solution parser. |

All training uses `open_instruct/grpo_fast.py`. The GRPO/data-loader options
below live on `StreamingDataLoaderConfig` / `DataLoaderConfig` in
`open_instruct/data_loader.py`.

## Setup

Same as upstream — see [`OPENINSTRUCT_README.md`](OPENINSTRUCT_README.md):

```bash
uv sync
```

## Running the experiments

Each script in `experiments/<name>/` is a thin wrapper that schedules a Beaker job
with `mason.py` and then runs `grpo_fast.py` inside it:

```bash
bash experiments/gsm8k/qwen2.5_0.5b_gsm8k.sh
```

- The Beaker fields are placeholders. Set them per run:
  `BEAKER_IMAGE=you/open_instruct WORKSPACE=... CLUSTER=... BUDGET=... bash experiments/gsm8k/qwen2.5_0.5b_gsm8k.sh`
- **Without Beaker:** delete the `uv run mason.py ... --` prefix and run the
  `uv run open_instruct/grpo_fast.py ...` body under your own launcher
  (`accelerate`/`torchrun`), keeping the flags unchanged.
- Extra CLI args are forwarded to `grpo_fast.py` (`"$@"`), which is how the paper
  sweeps were run.

See [`experiments/README.md`](experiments/README.md) for datasets and full
hyperparameters, and `scripts/data/rlvr/` for how the pass-rate datasets were
built.

## How the async never-give-up plumbing works

### 1. The async pipeline

`grpo_fast.py` runs, connected by Ray queues:

- **learners** (DeepSpeed), which consume one packed batch per training step;
- a pool of **vLLM inference actors** (`LLMRayActor`) that generate completions;
- a single **`DataPreparationActor`** with a background thread that turns finished
  generations into training batches;
- a **`weight_sync_thread`** that pushes updated policy weights to the inference
  actors.

Two queues carry the traffic: `prompt_Q` (prompts → inference) and
`inference_results_Q` (per-prompt `GenerationResult`s → data prep).

The data-prep thread prefills `async_steps × global_batch_size` prompts, then for
each step calls **`accumulate_inference_batches`**, which pulls `GenerationResult`s
off `inference_results_Q` until it has enough *accepted* prompt groups for one
batch, tokenizes/packs them, computes grouped advantages, and exposes them via
`get_data(step)` to the `StreamingDataLoader` on each learner.

- The thread stays up to `async_steps` ahead of the last consumed step, so
  training is off-policy by up to `async_steps` weight versions.
  `--async_steps 1` is the minimal setting.
- `--inflight_updates` lets a weight sync happen without first draining in-flight
  generations.
- `--sync_sampling` forces `async_steps == 1` and, instead of continuously
  replenishing `prompt_Q`, enqueues exactly one fresh batch of prompts per step —
  a deterministic per-step prompt set, close to on-policy.

### 2. Zero-std filtering

`--filter_zero_std_samples` (**on by default**; required by `--active_sampling`):
if all `num_samples_per_prompt_rollout` completions for a prompt get the same
reward (reward std 0 — the prompt is fully solved or fully failed), the group
gives GRPO no gradient signal and is dropped from the batch.

`--active_sampling`: keep pulling extra prompts (up to
`max_samples_multiplier×` the batch) until a full batch of non-zero-std groups is
assembled, so every training batch is full of informative groups.

### 3. Never-give-up retries

Rather than only dropping a zero-std group, requeue the **same prompt** for
another rollout at a later policy version:

- `--never_give_up <p>` — requeue each zero-std unsolved prompt with probability
  `p ∈ [0, 1]`; or
- `--never_give_up_int <n>` — requeue up to `n` times per prompt.
  (The two are mutually exclusive.)

Mechanics:

- Prompt ids are `"{epoch}_{index}"`; each retry appends `_1`, `_2`, … so ids
  don't collide. `get_never_give_up_chain_id` maps every retry back to its base
  id.
- `NeverGiveUpAccumulationState` (held by the `DataPreparationActor`,
  lock-guarded, checkpointed) carries per-chain state across steps: buffered
  `GenerationResult`s, running best reward, response/reward-sum counts, and
  attempt count.
- **Acceptance** (`--never_give_up_accept_on`): a later attempt is accepted if it
  has non-zero std, or — `better` (default) — its max reward beats the chain's
  previous best, or — `different` — its reward differs from the previous best.
- A chain whose best reward reaches `max_possible_score` stops retrying and is
  recorded as solved/filtered. If `--never_give_up_int` retries run out, the
  chain is *given up* and dropped (logged in `filtered/given_up_prompts_resamples`).
- `--maintain_pending_ngu_age <steps>` bounds how stale buffered completions may
  be (in policy versions) before they're discarded instead of merged.

### 4. Merging retries into one training group

When a retry is finally accepted:

- `--maintain_pending_ngu_completions` (**on by default**): all buffered attempts
  for the chain are concatenated (`merge_generation_results`) into one logical
  prompt group. The group can therefore be larger than
  `num_samples_per_prompt_rollout` and now spans a range of rewards.
- `--maintain_ngu_completions_downsample`: after merging, trim the group to an
  equal number of correct and incorrect completions.
- `--maintain_pending_ngu_counts` (+ `--ngu_count_baseline`): instead of / in
  addition to keeping tensors, keep just the count and reward-sum of discarded
  attempts and fold them into the GRPO advantage **baseline** (the mean/std
  denominator) via `prompt_baseline_sample_counts` / `prompt_baseline_reward_sums`
  in `compute_grouped_advantages`.
- `--maintain_pending_ngu_count_rescale {ratio,anchor_pos,count_ratio}`: rescale
  the resulting advantages to compensate for the inflated baseline count.
- `--advantage_normalization_type centered` (used by all paper runs): subtract the
  group mean, do **not** divide by group std.

### 5. Metrics and resume

- Histograms: `filtered/prompts_resamples*` and
  `filtered/given_up_prompts_resamples*` (1-based attempt numbers), alongside the
  standard per-dataset filtered / solved / zero-reward prompt and completion
  counts.
- `never_give_up_state` is saved with the `DataPreparationActor` state and
  restored on resume; `--ignore_resume_never_give_up_state` starts the retry
  chains fresh.
