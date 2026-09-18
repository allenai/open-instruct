# Terminal RL: an end-to-end walkthrough of `grpo_fast.py`

This document walks through one Terminal-RL training run from the point of view of
someone who knows RL but has not looked at how rollout engines and trainers are wired
together in this codebase. It answers, concretely: who computes which logprobs, where the
train/inference mismatch comes from, how async is layered on top of a synchronous loop,
how batches are formed on both sides, where the reference policy lives, and how weights
get from the trainer into vLLM.

It complements (and links into) the reference docs; read those when you need every
branch of the code, read this when you need the story:

- [grpo_fast_internals.md](grpo_fast_internals.md) — actor-by-actor reference
- [rollout_loop_internals.md](rollout_loop_internals.md) — token-level rollout loop reference
- [grpo_pipeline_overview.md](grpo_pipeline_overview.md) — plain-language overview and arg reference
- [monitoring_and_debugging_runs.md](monitoring_and_debugging_runs.md) — what the wandb metrics mean
- [podman_sandboxes.md](../podman_sandboxes.md) — how the podman sandbox backend is set up and plugs in

Throughout, the concrete example is the Qwen3.5-9B DPPO Terminal script
(`scripts/general_agent/terminal/rl/qwen35_9b_base_tmax_10k_dppo_32k.sh`). Its relevant
numbers:

| setting | value | meaning |
|---|---|---|
| `--num_learners_per_node 8 8` | 16 trainer GPUs on 2 nodes | |
| `--vllm_num_engines 16 --vllm_tensor_parallel_size 1` | 16 vLLM engines on the other 2 nodes | inference and training are on **disjoint GPUs** |
| `--sequence_parallel_size 4` | Ulysses SP=4 | 16 learners → 4 data-parallel (DP) groups |
| `--num_unique_prompts_rollout 32 --num_samples_per_prompt_rollout 8` | 32 groups × 8 = 256 trajectories per step | |
| `--async_steps 4` | generation runs up to 4 steps ahead of training | |
| `--response_length 32768 --per_turn_max_tokens 8192 --pack_length 35840` | trajectory / turn / packed-row token budgets | |
| `--max_steps 64` | max tool calls per trajectory | |
| `--pool_size 512` | sandbox actors | |
| `--use_vllm_logprobs true --loss_fn dppo --beta 0.0` | ratio against vLLM's logprobs, DPPO trust region, no KL term | |

File references below are to `open_instruct/…` and were checked against the `omni_agent`
branch in September 2026; line numbers drift, function names do not.

---

## 1. The cast

Everything is a Ray actor except the driver's main thread and one background thread.

| actor | count / placement | owns |
|---|---|---|
| `PolicyTrainerRayProcess` (`grpo_fast.py`) | 1 per learner GPU | DeepSpeed ZeRO-3 policy + optimizer; the **frozen reference policy** (if `load_ref_policy`); a `StreamingDataLoader`; one end of the NCCL weight-transfer group |
| `LLMRayActor` (`vllm_utils.py`) | 1 per inference GPU (× TP) | a vLLM async server; the multi-turn rollout loop; tool dispatch |
| `EnvironmentPool` + `SWERLVanilluxSandboxEnv` (`environments/`) | 1 pool, 512 env actors, CPU | each env actor drives one podman container at a time |
| `DataPreparationActor` (`data_loader.py`) | 1, CPU | rewards → advantages → packing → per-rank sharding; the async backpressure |
| `ActorManager` (`actor_manager.py`) | 1, CPU | the `should_stop` flag that pauses engines during weight sync; token/timing stats |
| main thread (`run_training`) | driver | the training loop |
| weight-sync thread (`weight_sync_thread`) | driver | copies trainer weights into vLLM after each step |

Two Ray queues connect them: `prompt_Q` (DataPrep → engines) and `inference_results_Q`
(engines → DataPrep). Both have capacity `(async_steps + 1) × num_unique_prompts_rollout`
plus the eval set.

---

## 2. One training step, synchronous mental model

Pretend for now that `async_steps` were 1 and everything happened in sequence. This is
not how the code runs, but it is the loop the async version is optimizing.

```python
# ---------------- ROLLOUT (16 vLLM GPUs) ----------------
for prompt in prompt_Q:                        # LLMRayActor._prefetch_worker
    for j in range(8):                         # add_request: n=8 fans out to 8 coroutines
        spawn process_request(prompt, seed + j)

async def process_request(prompt):             # vllm_utils.process_request — ONE trajectory
    sandbox = await pool.acquire_reset(task_id, image)    # fresh container
    tokens, logprobs, mask = [], [], []
    for turn in range(64):
        out = vllm.complete(prompt + tokens,
                            max_tokens=min(8192, 32768 - len(tokens), context_left),
                            logprobs_mode="processed_logprobs")
        tokens += out.ids; logprobs += out.logprobs; mask += [1] * len(out)   # model tokens: trained on
        calls = parser.parse(out.text)
        if not calls: break                                                    # final answer
        obs = await sandbox.step(calls[0])                                     # bash in the container
        obs_ids = tokenize(chat_template(role="tool", obs))
        tokens += obs_ids; logprobs += [0.0] * len(obs_ids); mask += [0] * len(obs_ids)  # not trained on
    reward = last per-turn reward (tests passed → 1.0)
    return tokens, logprobs, mask, reward, model_step

# 8 sibling trajectories → one GenerationResult → inference_results_Q

# ---------------- DATA PREP (1 CPU actor) ----------------
for group in inference_results_Q:              # group = 8 rollouts of one prompt
    if std(group.rewards) == 0: drop (and request a replacement prompt)
    advantage = rewards - mean(rewards)        # --advantage_normalization_type centered
pack trajectories into rows of <= 35840 tokens; shard rows across the 4 DP groups
prepared_data[step] = shards                   # trainers block on get_data(rank, step)

# ---------------- TRAIN (16 learner GPUs) ----------------
def step(training_step):                       # PolicyTrainerRayProcess.step
    rows = dataloader.next()                   # blocks on DataPrep
    ref_lp = forward(ref_policy, rows)         # no grad; only matters if beta > 0
    for row in rows:
        new_lp = forward(policy, row)          # with grad
        old_lp = row.vllm_logprobs             # --use_vllm_logprobs
        ratio  = exp(new_lp - old_lp)
        mask   = dppo_mask(new_lp, old_lp, adv, ratio, tv_threshold=0.1)
        loss   = sum(-adv * ratio * mask * response_mask) / global_token_count + beta * KL(new_lp, ref_lp)
        loss.backward()
    optimizer.step()                           # once per training step

# ---------------- WEIGHT SYNC (driver thread) ----------------
actor_manager.set_should_stop(True)           # engines stop dequeuing new prompts
trainers.broadcast_to_vllm()                   # ZeRO-3 all-gather → NCCL broadcast into vLLM
engines.set_model_step(training_step)
actor_manager.set_should_stop(False)
```

### 2.1 Who computes what

Nobody ships logits anywhere. Three different logprob evaluations happen per response
token:

| who | what | how it's used |
|---|---|---|
| vLLM, during sampling | `log μ(token)` for the sampled token only, after temperature (`logprobs_mode="processed_logprobs"`) | shipped to the trainer as `vllm_logprobs`; becomes `old_logprobs` when `use_vllm_logprobs` |
| trainer, policy forward | `log π_θ(token)` from the HF model under ZeRO-3 + SP, with grad | `new_logprobs` |
| trainer, ref forward | `log π_ref(token)`, no grad | KL term, only if `beta > 0` |

The vLLM logprobs are one float per token. The trainer recomputes the full forward
because it needs gradients, and because `π_θ` has moved since the rollout.

### 2.2 Where the train/inference mismatch comes from

`ratio = exp(new_lp - vllm_lp)` differs from 1 for two independent reasons:

1. **Numerics.** Same weights, different kernels: vLLM (bf16 paged attention, the Triton
   GDN prefill backend, prefix caching) vs. HF eager + DeepSpeed + Ulysses. Both sides
   patch the LM head to fp32 (`patch_vllm_qwen3_5_lm_head_fp32`,
   `patch_hf_lm_head_fp32`) to shrink this. Logged as `debug/vllm_local_logprob_diff_*`
   and `debug/vllm_vs_local_logprob_diff_reverse_kl`.
2. **Staleness.** With async, the rollout was sampled from weights at step `k−1…k−4`
   and trained on at step `k`. See §7.

With `--use_vllm_logprobs true`, `old_lp = vllm_lp`, so the ratio corrects for both at
once. Without it (and one mini-batch), `old_lp = new_lp.detach()` on the first pass,
ratio ≡ 1, the clip never fires, and you are doing REINFORCE-with-baseline on stale data.
Every Terminal script sets it to true for this reason.

### 2.3 Who hosts the reference policy

Each trainer GPU holds a second, frozen, ZeRO-3-sharded copy (`self.ref_policy`) and runs
one no-grad forward over its batch at the start of every `step()`. The ref never lives on
the vLLM side and is never synced. `--ref_policy_update_freq` + `--alpha` EMA-update it
if set.

`load_ref_policy` defaults to **True**. The 9B script sets `--beta 0.0` but does not pass
`--load_ref_policy false`, so it loads the ref and pays a ref forward per step for a KL
term multiplied by zero. Adding `--load_ref_policy false` frees that memory and time (the
config asserts `beta == 0` in that case).

---

## 3. The rollout loop in detail

### 3.1 From dataset row to vLLM request

```
DataPreparationActor                              LLMRayActor
  next(iter_dataloader) → row                       _prefetch_worker thread:
  add_prompt_to_generator(row) ───► prompt_Q ───►     while not should_stop and active < inference_batch_size:
    PromptRequest(                                       req = prompt_Q.get(); add_request(req)
      prompt = row["input_ids_prompt"],                    metadata[model_step] = current_model_step
      generation_config(n=8, temperature=1.0),             for j in range(8): spawn process_request(seed + j)
      index, prompt_id = f"{epoch}_{index}",
      env_config = base ⊕ row["env_config"])
```

- The prompt is **already tokenized with the chat template applied**
  (`dataset_transformation.rlvr_tokenize_v1`), including the tool schema from
  `get_tool_definitions()` and the `--system_prompt_override_file`. vLLM is hit via the
  `/completions` endpoint with raw token ids, never `/chat/completions`.
- `inference_batch_size` defaults to vLLM's KV-cache max concurrency
  (`LLMRayActor.get_kv_cache_info`), so in-flight trajectories per engine are capped by
  memory, not by a batch size.
- `data_loader._merge_env_config` merges the per-row `env_config` column over the CLI
  `--tool_configs`. That is how each trajectory learns its `task_id` and container image.

### 3.2 One trajectory (`vllm_utils.process_request`)

```python
sandbox = await pool.acquire_reset({"max_steps": 64, "task_id": ..., "image": ...})
tokens, logprobs, mask = [], [], []
prompt = list(original_prompt)
while rollout.step_count < 64 and not rollout.done:
    budget = min(32768 - len(tokens),              # --response_length
                 max_model_len - len(prompt),      # context left
                 8192)                             # --per_turn_max_tokens
    out = await client.completions.create(prompt=prompt, max_tokens=budget,
              extra_body={"return_token_ids": True, "cache_salt": base_request_id,
                          "logprobs_mode": "processed_logprobs", ...})
    tokens += out.token_ids; logprobs += out.logprobs; mask += [1] * len(out)   # MODEL tokens
    prompt += out.token_ids

    calls = tool_parser.parse_tool_calls(out.text).tool_calls ∩ allowed_tools
    if not calls: break                           # no tool call = final answer

    for tc in calls:                              # normally one `bash`
        rollout.step_count += 1
        res = await asyncio.wait_for(sandbox.step(tc), timeout=tool_call_timeout)
        rollout.rewards.append(res.reward); rollout.done |= res.done

    obs_ids = tokenize(tool_parser.format_tool_outputs([res.result], role="tool"))
    obs_ids, excess = truncate to fit max_model_len and the remaining response budget
    tokens += obs_ids; logprobs += [0.0] * len(obs_ids); mask += [0] * len(obs_ids)  # OBSERVATION tokens
    prompt += obs_ids
    if excess > 0: break                          # context full
finally:
    pool.release(sandbox)
```

Points worth internalizing:

- **Three lists, one length.** `tokens`, `logprobs`, `mask` stay in lockstep for the whole
  trajectory (prompt excluded). Model tokens: `mask=1`, real logprob. Tool output:
  `mask=0`, logprob `0.0` placeholder (`process_tool_tokens`). `--mask_tool_use` defaults
  True; off, the model would be trained to reproduce shell output.
- **The whole trajectory is one "response."** Downstream there is no per-turn structure
  except `mask` (and `dones`, used only by the value-model path).
- **Termination**, in the order checked: `rollout.done` (tests ran), 64 steps, response
  budget exhausted, context exhausted, no tool call parsed, or an observation overflowed
  the context. `finish_reason` is only the *last vLLM turn's* reason, which is why
  DataPrep defines truncated as `finish_reason != "stop" OR len(response) >= response_length`.
- **Format-error nudging** (`tool_call_format_error_feedback`) is off by default for
  this env, so a turn without a tool call simply ends the episode. Enabled, the model gets
  a `user`-role nudge instead and the loop continues.
- **Empty trajectories** (e.g. reset failed) get a single EOS with `logprob = NaN` so
  packing does not choke.
- `cache_salt=base_request_id` scopes vLLM's prefix cache to the 8 siblings of a prompt
  so they share the system prompt + task KV.

### 3.3 The sandbox side (`environments/swerl_vanillux_sandbox.py`, `environments/pool.py`)

- `EnvironmentPool` keeps 512 env actors in an asyncio queue. `acquire_reset` pops one
  (a timeout means an actor crashed without release), calls `reset(**kwargs)`, and
  rotates podman hosts on host-level failures.
- `reset` → `_do_reset`: resolve the image (`task_data/<task_id>/image.txt`), start a
  **fresh container**, upload the task files, install a bash wrapper. **Tests are not
  uploaded until submit** so the model cannot peek. Retries with backoff. With
  `SWERL_RESET_FAILURE_ZERO_REWARD=1` (set in the script) a final failure does not raise:
  the trajectory becomes a zero-token, reward-0 rollout, and it *does* enter the group's
  advantage computation as a 0.
- `step(bash)` → `_execute_bash`: run the command; observation = truncated stdout+stderr
  + `(exit_code=N)` (+ optional turns-remaining line). If output contains the submit
  marker → `_run_tests`: upload `/tests`, run `test.sh` under `test_timeout`, read
  `/logs/verifier/reward.txt`, clamp to [0, 1], return `done=True`. Sandbox OOM →
  `done=True, reward=0`.
- The rollout loop's `tool_call_timeout` is a separate outer timeout; on trip it appends
  reward 0 and an error observation and the episode continues.

### 3.4 From 8 trajectories to one scored group

`accumulate_completions` collects the 8 sibling outputs; `finalize_completed_request` →
`process_completed_request` builds a `GenerationResult` (responses, masks, logprobs,
finish_reasons, `request_info.rollout_states`, `model_step`) → `compute_rewards` →
`reward_fn` (`ground_truth_utils.RewardConfig.build`):

```python
verifier_score = PassthroughVerifier(...)  # env-only dataset: 0.0
turn_rewards = rollout_state["rewards"]    # e.g. [0, 0, 0, 1.0], one per tool call
turn_rewards[-1] += verifier_score * verification_reward
score = LastRewardAggregator(turn_rewards) # --reward_aggregator last → the test result
```

So the score is exactly the number in `reward.txt`, binary in practice.
`--verification_reward 1.0` is a no-op here. The result is put on `inference_results_Q`.

---

## 4. The DataPreparationActor

### 4.1 The loop

```python
def _data_preparation_loop(self):
    for _ in range(async_steps * 32): add_prompt_to_generator(next(iter_dataloader))   # prefill 128
    for step in range(training_step, num_training_steps):
        while step - last_consumed_step > async_steps: sleep(0.1)     # BACKPRESSURE
        result, batch, reward_metrics, stats = accumulate_inference_batches(num_prompts=32, ...)
        advantages = compute_group_advantages(scores, 8, "centered")   # A = r − mean(group)
        # optional: concave length penalty, evolving rubrics, truncation / non-submitting masking
        packed = pack_sequences(..., pack_length=35840, min_num_batches=dp_world_size)
        packed.advantages = per-token broadcast of the per-trajectory scalar
        collated = prepare_collated_data_for_workers(packed, dp_world_size=4, per_device_bs=1)
        with lock: prepared_data[step] = collated; metrics[step] = ...; current_prepared_step = step
```

### 4.2 `accumulate_inference_batches`: filtering and active sampling

Pulls groups **one at a time** from `inference_results_Q` until it has 32 *kept* groups:

1. **Stale drop.** If `training_step − result.model_step > async_steps`, drop the group
   (`stale_results_dropped`) and enqueue a replacement. This is the hard staleness cap.
2. **Replenish.** For every group consumed, kept or not, immediately enqueue
   `next(iter_dataloader)`. This keeps ~128 prompts queued regardless of filtering.
3. **`no_resampling_pass_rate`.** If the group's solve rate ≥ threshold,
   `iter_dataloader.exclude_index(index)`: the prompt is never drawn again this run.
4. **Zero-std filter** (`filter_zero_std_samples`, default True). All-pass or all-fail
   groups carry no advantage signal.
   - With `--active_sampling` (the script): the filtered group does **not** count toward
     32, so the loop keeps consuming until 32 informative groups exist. The training
     batch is always 32 groups; the number of rollouts generated per step is
     `32 / (1 − p_degenerate)`. This is why `async_steps > 1` is asserted with active sampling.
   - Without it: the group counts toward 32 and is dropped, so the batch shrinks
     (`real_batch_size_ratio < 1`).
   - Filtered groups are bucketed as `filtered_prompts_zero / _solved / _nonzero`;
     `val/avg_group_performance_pre_filter` re-includes them so you can see the true solve
     rate rather than the post-filter one.
5. If everything was filtered, the actor stores empty batches for the step; trainers see
   an empty batch and skip, but the step still counts.

### 4.3 Packing (`rl_utils.pack_sequences`)

Greedy first-fit in arrival order: append `prompt + response` to the current row; when
the next one would exceed `effective_pack_length`, flush. `min_num_batches = dp_world_size`
shrinks `effective_pack_length` to `total_tokens // 4` (capped at `pack_length`) so every
DP rank gets ≥ 1 row; a rank with no rows would hang the ZeRO-3 collectives. A 32k
trajectory never shares a row; short ones pack densely (`packed_ratio` = rows /
trajectories). `attention_masks` hold **segment ids** (1,1,1,2,2,…), from which
`position_ids` are reset per segment.

The per-token tensors that come out, all the same length as the row:

| field | contents |
|---|---|
| `query_responses` | concatenated `prompt + response` of several trajectories |
| `position_ids` | reset to 0 at each trajectory boundary |
| `response_masks` | 0 on prompt and tool-output tokens; `i+1` (trajectory index) on trained tokens |
| `advantages` | per-trajectory scalar broadcast via `response_masks` ids |
| `vllm_logprobs` | NaN on prompt tokens, 0.0 on tool tokens, real values on model tokens |
| `model_steps` | weight version that generated each token (enqueue-time step) |
| `rollout_sample_ids`, `prompt_masks`, `dones` | bookkeeping for sequence-level losses, traces, value model |

### 4.4 Sharding and the deadlock guarantee

`prepare_collated_data_for_workers` truncates the row count to a multiple of
`dp_world_size` (dropping stragglers with a warning), gives each DP rank a contiguous
slice of `B = total // 4` rows, shuffles within the rank, and collates into micro-batches
of `per_device_train_batch_size` (1). **Every rank gets exactly `B` rows.** This is the
actor's reason to exist: ZeRO-3 forward/backward are collectives, so ranks must execute
the same number of them or the job wedges. The 4 SP ranks of a DP group all call
`get_data(dp_rank, step)` and receive identical rows, then split them (§5.2).

### 4.5 `get_data`, idle metrics, checkpoint

- `StreamingDataLoader._iter_batches` on each trainer does
  `ray.get(actor.get_data(dp_rank, step))`; the actor spins until
  `current_prepared_step >= step`, records `last_consumed_step` (which releases the
  backpressure), and deletes steps `< step − 1`. The wait is logged as
  `time/trainer_idle_waiting_for_inference`; the mirror on the prep side is
  `time/generation_idle_waiting_for_trainer`. Together they tell you which side is the
  bottleneck.
- If the prep thread died, `get_data` re-raises into the trainer so the failure is loud.
- **Checkpoint/resume.** `get_state()` = `{training_step, last_consumed_step,
  iter_dataloader.state_dict()}`, saved with the DeepSpeed state every
  `--checkpoint_state_freq`. On resume the loop re-prefills 128 prompts from the restored
  position. Everything in `prompt_Q`, in flight in vLLM, or sitting in
  `inference_results_Q` at kill time is **lost and regenerated**, so a resume costs about
  `async_steps` steps of rollout compute. The `no_resampling_pass_rate` exclusion set is
  not in `HFDataLoader.state_dict`, so solved-out prompts come back after a resume.

---

## 5. The trainer step (`PolicyTrainerRayProcess.step`)

### 5.1 Arrival

`rows = next(self.dataloader)` returns the `CollatedBatchData` for this DP rank plus the
DataPrep metrics dict for the step. Everywhere below uses `[:, 1:]` slices because the
logit at position *t* predicts token *t+1*.

### 5.2 Sequence-parallel split (`utils.SequenceParallelSplitter.split_collated_batch`)

With SP=4, four learner GPUs jointly process the **same** row: all-gather the max length
across the 4 ranks, pad to a multiple of 4, and each rank slices its contiguous quarter of
every tensor. The un-split `position_ids` are stashed as `global_position_ids` so the
Qwen3.5 linear-attention (GDN) layers can build an FLA context-parallel context: the
conv/SSM state must flow across chunk boundaries. Attention uses Ulysses all-to-all inside
the patched model. Consequence: every per-token reduction (loss denominators, sequence
TIS masks, metrics) has to all-reduce across `self._sp_group`, which is why those calls
are sprinkled through the code.

### 5.3 Pre-passes

```python
accumulation_steps = ceil(num_rows / num_mini_batches − 0.5)   # num_mini_batches=1 → one optimizer step per training step
ref_logprobs = [forward(ref_policy, row) for row in rows]        # no grad, if load_ref_policy
old_logprobs = resolve_old_logprob(...)                          # = vllm_logprobs with use_vllm_logprobs
```

If `num_mini_batches > 1` (PPO-style multiple updates per batch), `old_logprobs` is
computed up front so later mini-batches ratio against the step-start policy.

### 5.4 The loss denominator

`--loss_denominator token` (default): `calculate_token_counts` counts response tokens
per row, **all-reduces across all 16 ranks**, and sums per accumulation group. Then

```python
loss = masked_mean(per_token_loss, response_mask, denominator=global_token_count)
loss *= world_size // sequence_parallel_size     # ×4: undo DeepSpeed's mean over DP ranks
```

Net: every trained token in the global batch has weight `1 / N_tokens`, independent of
packing or sharding. Long trajectories dominate the gradient (the "token-mean" choice from
DAPO). `sequence` gives each trajectory equal weight instead.

### 5.5 Per-row forward and loss

```python
for i, row in enumerate(rows):
    new_lp, entropy = forward_for_logprobs(model, row.ids, attention_mask=None, row.position_ids, temperature)
        # attention_mask=None → HF builds the block-diagonal mask from position_ids
        # logits[:, :-1] / temperature → log_softmax → gather label logprobs
    new_lp  = mask_logprobs(new_lp, response_mask)
    vllm_lp = mask_logprobs(row.vllm_logprobs[:, 1:], response_mask)
    old_lp  = vllm_lp                                              # use_vllm_logprobs
    ratio   = exp(new_lp − old_lp)                                 # π_θ / μ_behavior

    # optional trust-region / importance-sampling machinery, each None when disabled:
    tis_clamped = compute_tis_weights(old_lp, vllm_lp, cap)               # --truncated_importance_sampling_ratio_cap
    tis_mask    = compute_tis_mask(new_lp, vllm_lp, lower, upper)          # --tis_mask_lower/upper
    seq_tis     = compute_sequence_tis_mask(...)                           # --sequence_tis_mask_log_ratio_threshold
    dppo_mask   = compute_dppo_mask(new_lp, vllm_lp, adv, ratio, "tv", 0.1)  # --loss_fn dppo
    weights     = product of the non-None terms

    pg, pg2, pg_max, kl = compute_grpo_loss(new_lp, ratio, adv, ref_lp, config, tis_weights=weights)
    per_token = pg_max + beta * kl
    loss = masked_mean(per_token, response_mask, denom) * (world_size // sp_size)
    model.backward(loss)
    if (i + 1) % accumulation_steps == 0: model.step()
    dist.barrier(); torch.cuda.synchronize()                       # keep SP/ZeRO ranks in lockstep
```

**DAPO** (default `--loss_fn`): `max(−A·r, −A·clip(r, 1−ε_l, 1+ε_h))`. The max means the
clipped branch only bites when it would make the objective worse; `policy/clipfrac_avg`
is the fraction of tokens where it did.

**DPPO** (`grpo_utils.compute_dppo_mask`, `compute_binary_divergence`):

```
μ = exp(vllm_lp), π = exp(new_lp)              # prob of the *sampled* token under each policy
tv = |μ − π|                                    # Bernoulli TV over {sampled token, all others}
bad_high = (A > 0) & (r > 1) & (tv > δ)         # pushing UP a token that is already far up
bad_low  = (A < 0) & (r < 1) & (tv > δ)         # pushing DOWN a token that is already far down
M = ¬(bad_high | bad_low)
loss_t = −A · r · M                             # no clip; asymmetric masking replaces PPO's clip
```

Moves back toward the behavior policy are never masked. `debug/dppo_mask_frac_kept` is the
fraction of tokens that survived. DPPO requires `use_vllm_logprobs=True` so that the
anchor `μ` is the actual rollout policy.

**KL to the reference** (`beta > 0` only): `estimate_kl(new_lp − ref_lp, ratio)` returns
four estimators; `--kl_estimator` (default 2, the `expm1(−d) + d` form) enters the loss and
all four are logged as `objective/kl{0..3}_avg`. The gradient flows through `new_lp` only.

**Liger path** (`--use_liger_grpo_loss`, not used in the 9B script): `TiledGRPOLMHeadLoss`
recomputes the LM-head projection per tile so `[L, vocab]` logits are never materialized.

### 5.6 What goes back to the driver

Per-row `loss_stats_B` → `compute_metrics_from_loss_stats` → `local_metrics`
(`loss/policy_avg`, `policy/clipfrac_avg`, `val/ratio`, `val/ratio_var`, the KL
estimators, entropy if `--record_entropy`, the mask-kept fractions, `optim/grad_norm`,
`lr`, `_token_count`). `one_training_step` token-weights the averages across ranks using
`_token_count`; histograms are passed through from rank 0. DataPrep's per-step metrics
ride along in the same dict.

---

## 6. Weight sync

### 6.1 Setup, once

1. `from_pretrained` builds the ZeRO-3 engine for the policy (and the ref under an eval
   ZeRO config).
2. `dummy_optimizer_step`: a 2-token forward/backward with `loss × 0` and a real
   `optimizer.step()`. This forces ZeRO-3 to allocate partition pointers and NCCL
   communicators. It slightly advances Adam moments, which is harmless; then it
   invalidates ZeRO-3's parameter-prefetch trace so the real first step re-records one.
3. `setup_model_update_group`: trainer rank 0 plus all engine processes form a dedicated
   NCCL group of size `vllm_num_engines × TP + 1`. Trainer is rank 0; engine *i* is rank
   `i·TP + 1`. This uses vLLM's built-in `NCCLWeightTransferEngine`; ranks 1–15 of the
   trainer only hit the barrier.

### 6.2 Each sync (`weight_sync_thread`, `vllm_utils.broadcast_weights_to_vllm`)

```
main thread: one_training_step(k) returns → weight_sync_trigger.notify(k)
sync thread:
  actor_manager.set_should_stop(True)                   # engines stop DEQUEUING (in-flight continue)
  [trainer_i.broadcast_to_vllm() for all 16 ranks]      # collective: every rank must call
      rank 0: engine.sleep() on all engines → vLLM pause_generation(mode="keep")
      names/dtypes/shapes from ds_shape (no gather needed)
      rank 0: refs = [engine.update_weights(names, dtypes, shapes, packed=True)]   # engines block on NCCL recv
      with deepspeed.zero.GatheredParameters(all params):    # gather_whole_model=True: full model on every rank
          rank 0: clone each param contiguous → NCCLWeightTransferEngine.trainer_send_weights(packed=True)
  wait for the update_weights RPCs
  engine.wake_up() on all engines → resume_generation
  finally: set_should_stop(False); engine.set_model_step(k)
  push {time/weight_sync*} to weight_sync_metrics_Q (logged by the main thread next step)
```

- **Memory spike.** `gather_whole_model=True` materializes the full un-sharded model on
  every rank plus contiguous clones on rank 0. The per-parameter alternative
  (`gather_whole_model=False`) is slower but bounded; flip it for 70B.
- **Name mapping.** `_build_vlm_name_mapper` rewrites Qwen3.5's VLM-shaped HF names into
  what vLLM's loader expects; vLLM's `load_weights` handles fused QKV / gate-up merges.
- **`--inflight_updates true`** (the script): `sleep()` and `update_weights()` skip
  draining active requests, so trajectories mid-flight are paused, receive new weights,
  and resume. Two consequences: (1) one trajectory can contain tokens sampled from
  policies `k−1` and `k`; its `model_steps` record only the enqueue-time step, so the
  trainer cannot see this; (2) the paused requests' KV cache was computed with the old
  weights and is kept (`inflight_updates_recompute_kv_cache=False` default). Both are small
  extra sources of mismatch that the ratio/DPPO machinery absorbs. With
  `inflight_updates=False` you drain everything first: strictly on-version, but with 32k
  token, 64-turn trajectories that can stall the engines for minutes.
- `health_check_fn` in `run_training` crashes the run if this thread dies, because a dead
  sync thread means silently training on frozen engines.
- **Resume.** On resume the thread immediately syncs once so the engines start from the
  checkpoint's weights rather than the base model they loaded.

---

## 7. Now add async

Only three things change from §2:

1. DataPrep pre-fills `prompt_Q` with `async_steps × num_unique_prompts_rollout` prompts
   and tops it up one-for-one as groups are consumed (§4.2). Its loop refuses to prepare
   step `s` until step `s − async_steps` has been consumed by the trainers, which bounds
   memory and staleness.
2. Trainers and engines never wait on each other except at two points: `get_data` blocks
   a starved trainer; a full `prompt_Q` (or `should_stop`) blocks a starved engine. Weight
   sync happens between trainer steps while engines are mid-trajectory.
3. A step-`k` batch therefore contains trajectories generated by policies from steps
   roughly `k−4 … k−1` (`model_step_min / _max / _mean` are logged per step). Groups older
   than `async_steps` are dropped outright. This staleness is what the vLLM-logprob ratio
   and the DPPO mask exist to handle, and why `use_vllm_logprobs` is on for every
   Terminal run.

`async_steps = 1` is the minimum ("fully synchronous training is not supported"); even
then, generation for step `k+1` overlaps training of step `k`.

### 7.1 Reading the two idle timers

| `time/trainer_idle_waiting_for_inference` | `time/generation_idle_waiting_for_trainer` | meaning |
|---|---|---|
| high | ≈ 0 | generation-bound: add engines, shrink `response_length`, or raise `async_steps` |
| ≈ 0 | high | training-bound: add learners, or lower `async_steps` (it is only buying staleness) |
| both ≈ 0 | | balanced |

---

## 8. Quick reference: which flag touches which stage

| stage | flags |
|---|---|
| rollout budget | `response_length`, `per_turn_max_tokens`, `max_steps`, `tool_call_timeout`, `mask_tool_use` |
| sandbox | `tools`, `tool_configs`, `pool_size`, `backend_timeout`, `SWERL_*` env vars |
| reward | `reward_aggregator`, `verification_reward`, `add_concave_length_penalty`, `non_stop_penalty` |
| group formation | `num_unique_prompts_rollout`, `num_samples_per_prompt_rollout`, `filter_zero_std_samples`, `active_sampling`, `no_resampling_pass_rate`, `mask_truncated_completions`, `mask_non_submitting_completions` |
| advantage | `advantage_normalization_type`, `whiten_advantages` |
| packing / sharding | `pack_length`, `per_device_train_batch_size`, `sequence_parallel_size`, `num_learners_per_node` |
| loss | `loss_fn`, `clip_lower/higher`, `dppo_divergence_type/threshold`, `use_vllm_logprobs`, `truncated_importance_sampling_ratio_cap`, `tis_mask_*`, `sequence_tis_mask_*`, `loss_denominator`, `num_mini_batches`, `num_epochs` |
| KL / ref | `beta`, `load_ref_policy`, `kl_estimator`, `ref_policy_update_freq`, `alpha` |
| async / sync | `async_steps`, `inflight_updates`, `gather_whole_model`, `vllm_num_engines`, `vllm_tensor_parallel_size` |
