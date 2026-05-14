# `grpo_fast.py` — Internal Architecture Reference

This document describes the internal architecture of `open_instruct/grpo_fast.py`, the
DeepSpeed-based async GRPO trainer. It is intended as a deep-dive reference for people
reading or modifying the training code, not a user guide (see `docs/algorithms/grpo.md`
for that).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Ray Actors](#2-ray-actors)
3. [Data Flow End-to-End](#3-data-flow-end-to-end)
4. [Async Pipeline and Pipelining Depth](#4-async-pipeline-and-pipelining-depth)
5. [Weight Sync Mechanism](#5-weight-sync-mechanism)
6. [Training Loop (`run_training`)](#6-training-loop-run_training)
7. [Single Training Step (`one_training_step`)](#7-single-training-step-one_training_step)
8. [Trainer Step (`PolicyTrainerRayProcess.step`)](#8-trainer-step-policytrainerrayprocessstep)
9. [Loss Computation](#9-loss-computation)
10. [Advantage Computation](#10-advantage-computation)
11. [Metrics System](#11-metrics-system)
12. [Synchronization Points and Deadlock Prevention](#12-synchronization-points-and-deadlock-prevention)
13. [Key Configuration Parameters](#13-key-configuration-parameters)
14. [Key Files Map](#14-key-files-map)

---

## 1. System Overview

`grpo_fast.py` runs GRPO with sequence packing (DeepSpeed ZeRO) and an async pipeline
that overlaps inference (vLLM) with training (DeepSpeed). The system is orchestrated
with Ray and involves four distinct actor types:

```
┌────────────────────────────────────────────────────────────┐
│  Main Thread  (run_training / one_training_step)           │
│  - drives the training loop                                │
│  - calls .step() on all trainers, waits for results        │
│  - triggers weight syncs after each step                   │
└──────────┬─────────────────────────────────────────────────┘
           │
    ┌──────▼───────┐   NCCL broadcast   ┌──────────────────┐
    │  Trainers    │ ──────────────────► │  vLLM Engines    │
    │  (DeepSpeed, │                     │  (LLMRayActors)  │
    │  1 per GPU)  │                     │  - generate text │
    └──────────────┘                     │  - run tools     │
                                         └────────┬─────────┘
                                                  │ rollouts
                                         ┌────────▼─────────┐
                                         │  DataPrep Actor  │
                                         │  - pack seqs     │
                                         │  - compute advs  │
                                         │  - serve batches │
                                         └──────────────────┘
```

A separate **weight sync thread** (not a Ray actor) runs in the main process and handles
syncing trained weights from trainers to vLLM engines after each step.

---

## 2. Ray Actors

### 2.1 `PolicyTrainerRayProcess` (grpo_fast.py:188)

**`@ray.remote(num_gpus=1)`** — one per training GPU.

Owns:
- The DeepSpeed-wrapped policy model and optimizer
- An optional reference policy (for KL penalty)
- A `StreamingDataLoader` that pulls batches from `DataPreparationActor`
- The NCCL weight-transfer group for broadcasting to vLLM

Key methods:

| Method | Purpose |
|---|---|
| `from_pretrained()` | Initializes model, optimizer, dataloader on first call |
| `setup_model_update_group()` | Creates NCCL communicator group shared with vLLM engines |
| `broadcast_to_vllm()` | Gathers model weights and broadcasts them to all vLLM engines layer-by-layer |
| `step(training_step)` | Executes one training step; returns `(scalar_metrics, array_metrics)` |
| `update_ref_policy()` | EMA-updates the reference policy: `ref = α·param + (1-α)·ref` |
| `save_checkpoint_state()` | Saves model + optimizer + RNG + dataloader state for resumption |
| `save_model()` | Saves final model to HuggingFace format |
| `dummy_optimizer_step()` | Runs a zero-loss optimizer step to warm up ZeRO-3's NCCL communicators |

**ZeRO-3 initialization order matters:** `dummy_optimizer_step` must run before
`setup_model_update_group`. This is because ZeRO-3 needs at least one optimizer step
to allocate its internal parameter-partition pointers before NCCL broadcast groups can
be set up correctly.

### 2.2 `LLMRayActor` (vllm_utils.py)

Runs a vLLM async engine. Multiple engines can run on different GPU subsets.

- Pulls prompts from `prompt_Q` (a Ray queue)
- Generates completions with optional tool use (via `process_from_queue` worker thread)
- Pushes completed rollouts to `inference_results_Q`
- **Pauses generation** when `ActorManager.should_stop` is `True` (during weight sync)
- Resumes via `engine.wake_up()` after weight sync completes
- Tracks which "model step" (training iteration) its weights correspond to via `set_model_step()`

### 2.3 `DataPreparationActor` (data_loader.py:~1352)

Singleton Ray actor. Runs the `_data_preparation_loop` in a background thread.

Responsibilities:
- Consumes rollouts from `inference_results_Q`
- Filters zero-std-score groups (groups where all rollouts got the same score, so advantage is 0)
- Computes advantages per prompt group
- Packs multiple sequences into single training rows (sequence packing)
- Splits packed batches into per-rank sub-batches
- Stores `prepared_data[step]` and `metrics[step]` keyed by training step number
- Serves data via `get_data(rank, step)` RPC — blocks until the requested step is ready
- Cleans up old steps after they are consumed

**Active sampling:** When `active_sampling=True`, the DataPrep actor can re-enqueue
prompts that are difficult (low solve rate) for re-generation instead of discarding
them. Prompts above `no_resampling_pass_rate` are permanently excluded.

### 2.4 `ActorManager` (actor_manager.py)

Singleton Ray actor. The control plane.

- Owns the `_should_stop` flag (`set_should_stop(True/False)`)
- Aggregates token statistics and timing info from vLLM engines
- Hosts an optional HTTP dashboard for queue monitoring
- Reports per-training-step timing to the weight sync metrics queue

---

## 3. Data Flow End-to-End

```
1. Main thread calls add_prompt_to_generator(example) → puts prompt on prompt_Q

2. LLMRayActor dequeues prompt → runs vLLM inference (possibly with tool calls)
   → produces rollouts with reward scores → puts on inference_results_Q

3. DataPreparationActor._data_preparation_loop():
   a. Dequeues rollouts from inference_results_Q
   b. For each prompt group (num_samples_per_prompt_rollout rollouts):
      - Compute rewards (verifiable + rubric)
      - Filter if std(scores) == 0 (no learning signal)
      - Compute advantages (centered or standard normalization)
   c. Pack sequences: multiple rollouts concatenated into a single long sequence
      with position_ids reset at each sequence boundary
   d. Split into per-rank chunks
   e. Store in prepared_data[step] and metrics[step]

4. PolicyTrainerRayProcess.step():
   a. StreamingDataLoader calls ray.get(data_prep_actor.get_data(rank, step))
      → blocks until DataPrep has the batch ready
   b. Moves tensors to GPU
   c. Computes ref_logprobs via reference policy forward pass
   d. Computes new_logprobs via policy forward pass
   e. Computes GRPO loss, backpropagates
   f. Returns (scalar_metrics, array_metrics)

5. one_training_step():
   a. Collects results from all trainer ranks
   b. Token-weight-averages scalar metrics
   c. Logs to W&B (scalars as floats, arrays as Histograms)

6. weight_sync_thread (background):
   a. Pauses vLLM generation
   b. Broadcasts trained weights trainer → vLLM (NCCL)
   c. Resumes vLLM generation
   → vLLM now generates rollouts with updated weights for the next batch
```

---

## 4. Async Pipeline and Pipelining Depth

**`async_steps`** (default 8; set to 4 in dr-tulu script) controls how many training
steps worth of data the DataPrep actor generates ahead of the trainer.

Example with `async_steps=4`, `num_unique_prompts_rollout=2`:

```
Time →
Training:   [step 1 train] [step 2 train] [step 3 train] ...
DataPrep:   [step 1,2,3,4,5 prep running in background]
```

The queue capacity is `(async_steps + 1) × num_unique_prompts_rollout + num_eval_prompts`.
This limits how far ahead DataPrep can buffer, preventing unbounded memory use.

**Why this matters:** Without pipelining, the trainer would sit idle waiting for vLLM
to generate each batch. With pipelining, vLLM generates continuously and training is
nearly always compute-bound.

**Dataset size constraint:** The training dataset must have at least
`async_steps × num_unique_prompts_rollout` unique examples to avoid repeating prompts
within the pipeline buffer.

---

## 5. Weight Sync Mechanism

After each training step, trained weights must be copied from DeepSpeed trainer to
vLLM engines. This is handled by a dedicated background thread (not a Ray actor).

### `WeightSyncTrigger` (grpo_fast.py:~1613)

Thread-safe event + step counter. Main thread calls `notify(step)` after each
`one_training_step`. The weight sync thread blocks on `wait()`.

### `weight_sync_thread()` (grpo_fast.py:~1638)

Sequence:
1. `weight_sync_trigger.wait()` — blocks until main thread signals
2. `actor_manager.set_should_stop(True)` — vLLM actors stop dequeuing new prompts
3. `broadcast_to_vllm()` on each trainer rank:
   - Rank 0 all-gathers the full model (necessary for ZeRO-3 which shards weights)
   - Broadcasts each layer's weights to all vLLM engine processes via NCCL
   - Non-rank-0 trainers return empty lists (participate in all-gather only)
4. `engine.wake_up()` on all vLLM engines — unblocks their generation loop
5. `actor_manager.set_should_stop(False)` — allows new prompts to be dequeued
6. `engine.set_model_step(training_step)` — records which model version vLLM now has
7. Reports sync timing to `weight_sync_metrics_Q`

### `inflight_updates` flag

- **`True`** (default): After sending NCCL broadcast RPCs, immediately resume vLLM
  without waiting for RPCs to complete. Higher throughput; a few rollouts after sync
  may use slightly stale weights.
- **`False`**: Wait for all vLLM NCCL RPCs to confirm completion before unpausing.
  Strictly correct weights from the first rollout after sync, but slower.

---

## 6. Training Loop (`run_training`)

`run_training()` at grpo_fast.py:~2129 is the main loop:

```python
for training_step in range(resume_step, num_steps + 1):

    health_check_fn(weight_sync_thread_future)   # crash loudly if sync thread died

    # (optional) enqueue eval prompts if it's eval time

    data_thread_metrics = weight_sync_metrics_Q.get_nowait()  # timing from last sync

    num_step_tokens, episode = one_training_step(training_step, ...)

    num_total_tokens += num_step_tokens

    # (optional) checkpoint: save model, optimizer, dataloader state, data prep state

    weight_sync_trigger.notify(step=training_step)   # signal sync thread
```

**Resume from checkpoint:** DataPrep actor state, dataloader state (epoch position),
and model/optimizer state are all checkpointed together. This allows exact resumption
mid-training including correct dataset ordering.

---

## 7. Single Training Step (`one_training_step`)

grpo_fast.py:~1731

1. Dispatch `policy_group.models[i].step.remote(training_step)` to all N trainers in parallel
2. Wait via `ray_get_with_progress()` — blocks until ALL ranks return
3. Unpack `metrics, array_metrics = zip(*results)` — one dict per rank
4. (Optional) update reference policy if on schedule
5. Save checkpoint if on schedule
6. Aggregate metrics across ranks (see §11)
7. Log to W&B

The return value is `(num_step_tokens, episode)` — the main loop accumulates
`num_total_tokens` from this for throughput metrics.

---

## 8. Trainer Step (`PolicyTrainerRayProcess.step`)

grpo_fast.py:~588. Runs on each trainer GPU independently.

```
step(training_step):
    1. batch_data = next(self.dataloader)
       └─ Blocks on ray.get(data_prep_actor.get_data(rank, step))

    2. (optional) Split batch for sequence parallelism (Ulysses)

    3. Move tensors to GPU

    4. Compute ref_logprobs via forward pass through ref policy (no grad)

    5. If num_mini_batches > 1:
       Compute old_logprobs via forward pass through current policy (no grad)
       — needed for multi-epoch importance sampling ratio

    6. For each epoch (num_mini_batches):
       For each sample i in batch:
         a. Compute new_logprobs via forward pass (with grad)
         b. Mask logprobs to response tokens only
         c. Compute importance sampling ratio: ratio = exp(new_lp - old_lp)
         d. (optional) Compute TIS mask (trust-region gate)
         e. Compute GRPO loss (see §9)
         f. loss.backward() with DeepSpeed
         g. On accumulation boundary: optimizer.step()
         h. Record per-sample loss stats

    7. Aggregate loss stats into local_metrics

    8. Merge batch_metrics (from DataPrep) into local_metrics

    9. Return (local_metrics.get_metrics_list(), array_metrics)
```

**Gradient accumulation:** `accumulation_steps = ceil(num_samples / num_mini_batches)`.
`model.set_gradient_accumulation_boundary(is_boundary)` tells DeepSpeed when to
actually apply the accumulated gradient.

---

## 9. Loss Computation

grpo_utils.py:~406, model_utils.py:~795

### GRPO Loss (`compute_grpo_loss`)

Two loss variants, selected by `--loss_fn`:

**DAPO** (default):
```
pg_loss  = -advantage × ratio
pg_loss2 = -advantage × clamp(ratio, 1 - clip_lower, 1 + clip_higher)
policy_loss = mean(max(pg_loss, pg_loss2))    # take the larger (less negative) loss
```
This is the clipped surrogate objective from PPO, adapted for GRPO.

**CISPO:**
```
pg_loss = -advantage × clamp(ratio.detach(), max=1 + clip_higher) × new_logprobs
```
One-sided clip, no lower bound; loss passes gradient through `new_logprobs` directly.

**TIS (Truncated Importance Sampling) weights** (when `truncated_importance_sampling_ratio_cap > 0`):
Multiply `pg_loss` by `min(ratio, cap) / ratio` to cap extreme ratios. Reduces
variance from large off-policy gaps.

**TIS mask** (ScaleRL-style, `--tis_mask_lower` / `--tis_mask_upper`):
Binary mask zeroing out tokens where `ratio` falls outside `[1-ε_l, 1+ε_h]`. Prevents
gradient flow on tokens too far from the rollout policy.

**Total loss:**
```
total_loss = policy_loss + β × KL
```

### KL Estimators (`estimate_kl`, model_utils.py:~795)

Four estimators are always computed (shape `[4, B, T]`); `--kl_estimator` selects
which one enters the loss. All four are logged as `objective/kl{0,1,2,3}_avg`.

| Index | Formula | Notes |
|---|---|---|
| 0 | `new_lp - ref_lp` | Linear approximation |
| 1 | `(new_lp - ref_lp)² / 2` | Quadratic (always positive) |
| 2 | `expm1(-(new_lp - ref_lp)) + (new_lp - ref_lp)` | Numerically stable; preferred default |
| 3 | `ratio × (new_lp - ref_lp)` | Importance-weighted |

`new_lp - ref_lp` is clamped to `[-40, 40]` before all estimator computations.

**Old logprobs vs vLLM logprobs:** By default, `old_logprobs` (the denominator of the
IS ratio) is the current policy's logprobs on the first epoch (detached). With
`--use_vllm_logprobs`, the vLLM generation logprobs are used instead, which is a
stronger off-policy correction but incompatible with TIS capping.

---

## 10. Advantage Computation

data_loader.py:~1604

Advantages are computed per prompt group (all `num_samples_per_prompt_rollout` rollouts
for one prompt), then broadcast back to each rollout in the group.

```python
scores_per_prompt = scores.reshape(-1, num_samples_per_prompt_rollout)
mean_grouped_rewards = scores_per_prompt.mean(axis=-1).repeat(K)
std_grouped_rewards  = scores_per_prompt.std(axis=-1).repeat(K)

# --advantage_normalization_type centered (default):
advantages = scores - mean_grouped_rewards

# --advantage_normalization_type standard:
advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
```

**Why centered (not standard)?** Centered normalization preserves the scale of the
advantage signal. If a group has 3/4 correct, the wrong one gets `advantage = -0.75`
and correct ones get `+0.25`. Standard normalization would rescale these, obscuring the
absolute difficulty of the task.

**Filtering:** Groups where `std(scores) == 0` are dropped entirely. These groups
provide no gradient signal (all advantages = 0 after centering) and waste compute.
The metrics `batch/filtered_prompts_zero` and `batch/filtered_prompts_solved` track how
many groups were dropped and whether they were all-wrong vs all-correct.

**Packing advantages:** After packing, advantages are stored as a lookup array indexed
by a packed position mask. Position 0 in the lookup is reserved as 0 (for prompt
tokens). Response tokens get the group's advantage value; prompt tokens get 0.

---

## 11. Metrics System

### Scalar metrics (per-rank)

Each trainer returns a list of scalar dicts from `local_metrics.get_metrics_list()`.
One dict per training step (normally one dict, but multiple if gradient accumulation
spans multiple optimizer steps).

The main thread aggregates across ranks using **token-weighted averaging** for loss
metrics (each rank's contribution is weighted by how many tokens it processed) and
**simple averaging** for other metrics.

Token-weighted metrics:
`objective/kl{0,1,2,3}_avg`, `loss/kl_avg`, `loss/policy_avg`, `loss/total_avg`,
`policy/clipfrac_avg`, `policy/entropy_avg`, `val/ratio`, `val/ratio_var`

### Array metrics (from DataPrep, rank-0 only)

Arrays are passed through from `array_metrics[0]` (assumed identical across ranks since
all ranks get the same batch). They include:

| Key | Content |
|---|---|
| `batch/prompt_lengths` | Token counts for each prompt |
| `batch/response_lengths` | Token counts for each response |
| `val/advantages_hist` | Raw advantage values (one per rollout) |
| `val/solve_rate_hist` | Per-prompt solve rate (one per unique prompt) |
| `val/sequence_lengths_unsolved_hist` | Response lengths for unsolved rollouts |
| `val/sequence_lengths_solved_hist` | Response lengths for solved rollouts |

In W&B, arrays are logged as `wandb.Histogram`. **Edge case:** If all values in an
array are identical (zero range), numpy cannot create 64 histogram bins and raises
`ValueError`. The code guards against this by checking `np.ptp(arr) == 0` and falling
back to logging the scalar mean.

### KL metrics interpretation

`objective/kl{0,1,2,3}_avg` are all four KL estimators logged for comparison. Only the
one selected by `--kl_estimator` (default 2) actually contributes to training loss.
If all four show 0.00 it means either `beta=0.0` or the reference policy has identical
weights to the current policy.

---

## 12. Synchronization Points and Deadlock Prevention

There are five places where threads/actors block waiting for each other:

| Location | Blocks on | Can deadlock if... |
|---|---|---|
| `StreamingDataLoader._iter_batches()` | `data_prep_actor.get_data(rank, step)` | DataPrep is stuck waiting for vLLM |
| `DataPreparationActor._data_preparation_loop()` | `inference_results_Q` | vLLM paused for weight sync and DataPrep is still pulling |
| `weight_sync_trigger.wait()` | main thread `notify(step)` | main thread crashed before signaling |
| `ray_get_with_progress()` in `one_training_step` | all trainer `.step()` calls | one trainer rank hangs |
| `health_check_fn()` at loop top | weight sync thread health | sync thread crashed |

**Deadlock prevention:** The `should_stop` flag is the key mechanism. When weight sync
starts, `actor_manager.set_should_stop(True)` stops vLLM actors from dequeuing new
prompts. But they finish in-flight completions first, allowing DataPrep to drain.
DataPrep's `async_steps` buffer ensures it has work to give trainers even when vLLM is
paused.

**Over-buffering prevention:** `DataPreparationActor` checks that `step <=
_last_consumed_step + async_steps` before preparing data for a new step. This prevents
it from generating more than `async_steps` steps ahead, bounding memory use.

---

## 13. Key Configuration Parameters

From `GRPOExperimentConfig` (grpo_utils.py) and `StreamingConfig` (data_loader.py):

### Training algorithm

| Flag | Default | Effect |
|---|---|---|
| `--beta` | 0.05 | KL penalty coefficient. 0.0 = no KL constraint. |
| `--kl_estimator` | 2 | Which of the 4 KL estimators to use in the loss |
| `--clip_lower` | 0.2 | PPO clip lower bound (ratio ≥ 1 - clip_lower) |
| `--clip_higher` | 0.272 | PPO clip upper bound (ratio ≤ 1 + clip_higher) |
| `--loss_fn` | `dapo` | Loss variant: `dapo` (clipped surrogate) or `cispo` (one-sided) |
| `--advantage_normalization_type` | `centered` | `centered` = subtract mean; `standard` = also divide by std |
| `--truncated_importance_sampling_ratio_cap` | 2.0 | Cap IS ratio at this value. 0 = disabled. |

### Rollout and batching

| Flag | Default | Effect |
|---|---|---|
| `--num_unique_prompts_rollout` | 16 | Unique prompts per generation batch |
| `--num_samples_per_prompt_rollout` | 4 | Completions per prompt |
| `--async_steps` | 8 | Pipeline depth (steps DataPrep works ahead) |
| `--response_length` | 256 | Max tokens per response |
| `--pack_length` | 512 | Max tokens in a packed training sequence |
| `--temperature` | 0.7 | vLLM sampling temperature |

### Infrastructure

| Flag | Default | Effect |
|---|---|---|
| `--inflight_updates` | True | Don't wait for vLLM NCCL RPCs before resuming generation |
| `--deepspeed_stage` | 0 | ZeRO stage (3 = full weight sharding across GPUs) |
| `--num_learners_per_node` | [1] | Trainer processes per node (list, one per node) |
| `--vllm_num_engines` | — | Number of vLLM engine processes |
| `--sequence_parallel_size` | 1 | Ulysses sequence parallelism across trainers |
| `--load_ref_policy` | True | Load and use reference model for KL penalty |
| `--ref_policy_update_freq` | None | Steps between EMA reference policy updates |

### Active sampling / filtering

| Flag | Default | Effect |
|---|---|---|
| `--active_sampling` | False | Re-enqueue difficult prompts for more rollouts |
| `--filter_zero_std_samples` | True | Drop groups where all rollouts scored the same |
| `--no_resampling_pass_rate` | None | Exclude prompts above this solve rate from resampling |

---

## 14. Key Files Map

| File | Role |
|---|---|
| `open_instruct/grpo_fast.py` | Main orchestration: all Ray actors except vLLM and DataPrep, training loop, weight sync thread |
| `open_instruct/grpo_utils.py` | `GRPOExperimentConfig`, loss computation, KL estimators, loss stat helpers, sequence packing utilities |
| `open_instruct/data_loader.py` | `DataPreparationActor`, `StreamingDataLoader`, advantage computation, rollout accumulation, sequence packing |
| `open_instruct/actor_manager.py` | `ActorManager`: `should_stop` flag, token statistics, timing, dashboard |
| `open_instruct/vllm_utils.py` | `LLMRayActor`, tool call handling, vLLM async engine wrapper |
| `open_instruct/model_utils.py` | `estimate_kl`, forward pass utilities, model loading helpers |
| `open_instruct/evolving_rubric_step.py` | DR-Tulu evolving rubric generation and update logic |
| `open_instruct/data_types.py` | Shared dataclasses: `CollatedBatchData`, `BatchStatistics`, `PackedSequences` |
| `open_instruct/rl_utils.py` | `masked_mean`, rollout saving, token stat helpers |
| `scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh` | DR-Tulu experiment script (Qwen3.5-4B, tool use, evolving rubrics) |
| `scripts/train/debug/single_gpu_on_beaker.sh` | Minimal single-GPU debug script (~8 min) |
| `scripts/train/debug/large_test_script.sh` | Two-node debug script (~32 min) |
