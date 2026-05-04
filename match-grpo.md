# Matching grpo.py to grpo_fast.py — Current Status

Goal: get `grpo.py` (OLMo-core trainer) to learn equivalently to `grpo_fast.py` (HF / DeepSpeed trainer) on the Qwen3-4B-Base DAPO-Math setup.

## Reference runs (current)

| | run | wandb | AIME pass@1 |
|---|---|---|---:|
| Reference grpo_fast (1000 steps) | beaker `01KPTSPMH4Q2SBQMKGZ6YA145Q` | `parozgke` | 0.2156 |
| Latest grpo.py (1000 steps) | beaker `01KQJQQK0AGJZM0R5VF1JR0H8G` | `go8wry34` | 0.1948 |

Script under test: `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh`. Same hyperparameters as the grpo_fast launch except for trainer choice (`--fsdp_shard_degree 4 --fsdp_num_replicas 1` vs `--deepspeed_stage 2`).

## Bugs 1–4 (fixed, on this branch)

Documented in earlier revisions of this doc; all landed in `8300ff103` and ancestors:

1. Launch script not consuming the freshly-built image (`833f8dc`).
2. `inflight_updates` kwarg passed to a callback without that field (`0f04b02`).
3. **Optimizer never stepping.** `optim_step` / `zero_grads` overrides made `train_batch`'s call to `self.optim_step()` a no-op. Fixed by calling `super().optim_step()` from inside `train_batch` (`008dc95`).
4. `--lr_scheduler_type constant` silently routed to `LinearWithWarmup(alpha_f=0)`. Fixed by explicit routing + raising on unknown values (`008dc95`).

## Bug 5 (status: FULLY FIXED — single-doc and packed/doc_lens both correct)

Original finding (commit `05a966756`): single-doc forward through OLMo-core's Qwen3 port diverged from HF starting at `block_6` (max ≈288), causing the trainer↔vLLM TIS gap.

Re-ran the parity probe (`scripts/diagnostics/olmo_core_hf_parity.py`, restored from `939a1f631`) against the current pin `61091dba` on Beaker `01KQSYMVPRBNTG22WQXY4Y6TVM` then again on `01KQT1HM575JQREGW41Q55CKTB` (single GPU, ~1 min wallclock).

### Single-doc (no packing) — FIXED

| layer | max\|Δ\| (was) | max\|Δ\| (now) |
|---|---:|---:|
| `block_0`–`block_5` | ≤0.5 | ≤0.13 |
| **`block_6`** | **288** | **0.156** |
| `block_7`–`block_22` | ~288 | 0.13–0.25 |
| `block_23`–`block_35` | ~288 | 0.5–4.0 |
| `final_norm` | 3.5 | 1.0 |
| **logprob max\|Δ\|** | 3.1e-2 | **9.3e-3** |
| **logprob mean\|Δ\|** | 7.0e-3 | 3.3e-3 |
| argmax-agree | 100% | 100% |

The original `block_6` jump is gone. Per-block diffs grow gradually from bf16 noise (~1e-2) at block_0 to ~4 at block_35 — consistent with 36 stacked bf16 matmuls and acceptable for our purposes. **Bug 5 single-doc is fixed.**

### Packed-2-doc — FIXED (after probe was corrected)

Initial probe v1 (`01KQSYMVPRBNTG22WQXY4Y6TVM`) reported `packed FIXED block_6 max|Δ| = 5405`. That number was an artifact of an apples-to-oranges comparison: the HF baseline ran with `attention_mask=None`, so HF allowed cross-doc attention while OLMo-core (with `doc_lens`) blocked it. The whole 5400 was just "what one extra row of cross-doc attention contributes per layer, compounded 6 times."

Probe v2 (`01KQT1HM575JQREGW41Q55CKTB`, commit `bdccbdb7a`) builds a 4D additive `attention_mask` for HF that blocks cross-doc attention too. Now both sides have the same masking semantics:

| | BROKEN (HF intra-doc, OLMo no doc_lens) | FIXED (both intra-doc) |
|---|---:|---:|
| `block_0` max\|Δ\| | 5.48 | 7.8e-3 |
| `block_5` max\|Δ\| | 46.6 | 0.125 |
| **`block_6` max\|Δ\|** | 5405 | **0.160** |
| `block_35` max\|Δ\| | 806 | 8.0 |
| `final_norm` max\|Δ\| | 48.3 | 2.0 |
| logprob max\|Δ\| | 7.19 | **0.219** |
| logprob mean\|Δ\| | 0.97 | **0.029** |
| argmax-agree | 68.2% | **100%** |

**Packed FIXED tracks single-doc almost exactly** (block_6: 0.156 vs 0.160; block_35: 4.0 vs 8.0). The `doc_lens` path in OLMo-core works. The 700× regression we thought we saw was the probe's HF baseline being wrong, not olmo-core.

What this means:
- `Transformer.forward(..., doc_lens=..., max_doc_lens=...)` at pin `61091dba` is correctly plumbed through `_prepare_inputs` → `cu_doc_lens` → block kwargs → `Attention.forward` → `RotaryEmbedding.forward` (intra-doc RoPE, `rope.py:533-544`) → flash-attn-2 varlen (`backend.py:226-231`). All verified by reading source.
- The trainer passing `doc_lens` on every step is fine. **`doc_lens` is not the cause of the AIME regression or the 50× grad_norm gap.**
- The `BROKEN` column (no doc_lens) shows what dropping the kwarg costs: cross-doc attention bleed compounds to ~5400 by block_6. That confirms doc_lens is doing real work; we just shouldn't read its diff against an unmasked HF baseline.

## New observation: 50× grad_norm gap (current branch tip `8181927c4`)

Comparing the latest grpo.py run to the grpo_fast.py reference, aligned on `training_step`:

| metric | grpo.py (go8wry34) | grpo_fast (parozgke) | ratio |
|---|---:|---:|---:|
| `optim/grad_norm` mean | 0.0160 | 0.9165 | **57×** |
| `optim/grad_norm` median | 0.0143 | 0.8640 | **60×** |
| `loss/policy_avg` (late) | 0.30 | 0.37 | 0.81× |
| `lr` | 1e-6 | 1e-6 | 1.0× |
| `objective/verifiable_correct_rate` (late) | 0.54 | 0.49 | — train is **higher** in grpo.py |
| `val/sequence_lengths` (late) | 3826 | 4080 | 0.94× |
| `debug/vllm_vs_local_logprob_diff_mean` | ~6e-3 | ~6e-3 | 1.0× |

So grpo.py reports a per-step gradient norm ~50× smaller than grpo_fast.py from step 1 onward (new step-1 grad_norm 0.037 vs ref 2.00, with identical starting weights), even though loss values are within 20% of each other. AIME generalization is worse despite higher train reward — consistent with under-training.

### Loss-scaling hypothesis — ruled out

I checked whether grpo.py's loss path has a missing `* world_size` factor or a wrong-group token-count all-reduce.

- `train_module.dp_process_group` (olmo-core `train_module.py:202-203`) returns `get_dp_process_group(world_mesh)`, which **flattens both `dp_replicate` and `dp_shard` dims** (`olmo_core/distributed/parallel/__init__.py:403-417`). With `shard=4, replicas=1`, this group has size 4, not 1.
- Therefore `calculate_token_counts`'s `dist.all_reduce(..., group=dp_process_group)` reduces over all 4 shard ranks → `loss_denominator` is the global token count for that accumulation group, matching grpo_fast's default-group all-reduce.
- `loss * dp_world_size` = `loss * 4`, matching `grpo_fast.py:749`'s `loss * (world_size // sequence_parallel_size) = loss * 4`.
- HSDP's `_clip_grad_norm` (`train_module.py:611-638`) uses `nn.utils.get_total_norm` then `.full_tensor()` on the resulting DTensor — globally reduced, not per-shard.

Conclusion: loss path math is structurally equivalent to grpo_fast.py. The 50× grad_norm gap is **not** from loss scaling.

### Open hypotheses for the grad_norm gap

(Bug 5 / `doc_lens` ruled out by probe v2 above. Loss-scaling ruled out earlier.)

1. **DTensor norm aggregation under HSDP.** `get_total_norm` on `_NormPartial` DTensors with subsequent `.full_tensor()` should sum `p^norm` across shards before taking the root. A bug here would systematically under-report by ~√shard_degree (i.e. ~2×) — that's ~10× too small to explain a 50× gap on its own, but could combine with another factor. Easy to rule in/out by computing the norm manually after `loss.backward()`.
2. **`masked_mean` denominator differs from grpo_fast.** grpo.py uses `masked_mean(pg_loss + beta*kl, mask, None, loss_denominator)` where `loss_denominator` is the all-reduced token count over the dp_process_group. grpo_fast uses the same call but the denominator is reduced over the default global group. Need to verify these denominators are numerically equal in a real step (e.g. add `dist.barrier(); print(loss_denominator)` and compare).
3. **Activation checkpointing (`activation_memory_budget=0.5`).** Selective AC shouldn't change gradients, but it's a config delta. Try `activation_memory_budget=1.0` for a few steps and check grad_norm.
4. **Optimizer-state lifecycle.** `super().optim_step()` was wired in `008dc95` as the fix for Bug 3. Worth re-confirming the optimizer actually advances every accumulation boundary in a current run (e.g. by logging param hash before/after on rank 0 for a few steps).
5. **Gradient accumulation / micro-batching boundary.** grpo.py's accumulation loop in `train_batch` may divide gradients differently from grpo_fast's. If grpo.py is averaging across accumulation steps where grpo_fast is summing (or vice versa), grad_norm could differ by `num_accum_steps` (often ~16-64×).
6. **Reduce-scatter vs all-reduce averaging.** FSDP reduce-scatter divides by world_size in the gradient reduction; DeepSpeed Stage 2 all-reduce sums then divides by world_size. The math is equivalent IF the local pre-reduce gradient is the same — but if grpo.py's loss is *already* divided by world_size before backward (and grpo_fast's isn't, or vice versa), grad_norm differs by world_size (8× or 16×).

## What we know is fine

- `lr` matches across runs at every checkpoint.
- `debug/vllm_vs_local_logprob_diff_mean ≈ 6e-3` in both runs — the trainer's forward agrees with vLLM to the same degree in both trainers, so the *trainer↔vLLM* logprob path isn't the culprit (this metric does **not** test trainer-vs-trainer agreement, only trainer-vs-vllm).
- `val/ratio` and `val/tis_clipfrac` are not diagnostic with `num_mini_batches=1, num_epochs=1` (old logprob = new logprob by construction).
- Bug 5 update from earlier revisions: HF's Qwen3 with `attention_mask=None + position_ids` does **not** build an intra-doc mask — `position_ids` only affect RoPE. So grpo_fast.py's training-time HF forward also has cross-doc attention bleed, which is why the original packed-vs-solo probe didn't reveal a clean structural difference. The dominant issue was always the single-doc block_6 divergence.

## Recommended next moves

In rough priority order (Bug 5 closed; doc_lens ablation no longer the lead):

1. **Manual grad-norm check.** Single-rank instrumented variant that, after `loss.backward()` and before clip, computes `sqrt(sum(p.grad.detach().to_local().pow(2).sum() for p in params))` (DTensor-aware) and prints alongside the clip-reported value. If they disagree, the bug is in the DTensor `_NormPartial` aggregation; if they agree, the gradients themselves really are 50× smaller and we look upstream of clip.
2. **Bisect FSDP vs DDP under HSDP.** Run grpo.py with `fsdp_shard_degree=1, fsdp_num_replicas=8` (pure replica / DDP-like). If grad_norm jumps to ~1.0, the gap is FSDP-shard-specific (DTensor aggregation, reduce-scatter semantics). If it stays at ~0.02, the issue is in the loss path or accumulation, not sharding.
3. **Print `loss_denominator` and `loss` value at step 1.** Compare side-by-side to grpo_fast.py's `total_loss_token_count` and final `loss` at step 1 (same data, same starting weights). If `loss_denominator` differs, the masked_mean call is the culprit; if `loss` matches but grad_norm doesn't, it's downstream of `.backward()`.
4. **Confirm optimizer is stepping.** Log a fixed parameter's `abs().mean()` before and after a known step in both trainers. Rules out a subtler regression of Bug 3.
5. **Disable activation checkpointing** (`activation_memory_budget=1.0`) for a few steps; check if grad_norm changes. Should be a no-op for gradients; if it isn't, we have a recompute determinism problem.

## Probe + scripts

- `scripts/diagnostics/olmo_core_hf_parity.py` — single-rank HF↔OLMo-core forward + per-layer hidden-state diff probe.
- `scripts/diagnostics/launch_olmo_core_hf_parity.sh` — Beaker single-GPU launch wrapper. Run via `./scripts/train/build_image_and_launch.sh scripts/diagnostics/launch_olmo_core_hf_parity.sh`.

## Step-1 capture probe (new)

`open_instruct/_step1_capture.py` writes per-rank dumps to `$OPEN_INSTRUCT_DUMP_DIR/{trainer}_step1_rank{R}.pt` when `OPEN_INSTRUCT_DUMP_STEP=1`. Captured for both trainers on Qwen3-4B-Base DAPO with seed=1, identical hyperparameters except `--fsdp_shard_degree 4` vs `--deepspeed_stage 2`.

| run | exp | dump dir |
|---|---|---|
| grpo.py oc v3 | `01KQT919T937TN54MHAJPZ946X` | `/weka/oe-adapt-default/finbarrt/step1_capture/qwen3_4b_dapo_grpooc_step1cap_v3_20260504_145109/` |
| grpo_fast v5 | `01KQTD7TVQBAMQXXFJ5ZEM3QMP` | `/weka/oe-adapt-default/finbarrt/step1_capture/qwen3_4b_dapo_grpofast_step1cap_v5_20260504_151018/` |

Diff job `01KQTE27NWT6Y4G3VN3N2N87T2` (`scripts/train/qwen/diff_step1_dumps.py`) on rank0 dumps:

### Findings

1. **`response_masks` dtype mismatch**: oc=`torch.int64`, fast=`torch.bool`. Same shape contract; meaning identical, but if any downstream code does `mask.sum()` it's a type-coerce difference. Worth confirming masked_mean denominators are identical between trainers.

2. **`param_grads` dict empty for grpo_fast (399 vs 0)**. `snapshot_param_grads` iterates `model.named_parameters()` and reads `p.grad`. In **DeepSpeed Stage 2** the gradients are partitioned and stored on the optimizer (`engine.optimizer.averaged_gradients` / `single_partition_of_fp32_groups`), not on `p.grad`, so our naive collector saw zero gradients. The OLMo-core/HSDP side correctly sees 399 DTensor `p.grad`s (`is_dtensor=True`).

   - This means the rank0 dumps **cannot be diffed for gradients yet**. Need to extend `snapshot_param_grads` to pull from `engine.optimizer.averaged_gradients` for the deepspeed path before relaunching grpo_fast.

3. **Inputs payloads otherwise structurally identical**: same set of fields (`advantages`, `attention_masks`, `position_ids`, `query_responses`, `response_masks`, `vllm_logprobs`), all `list[Tensor]` of length 2, same dtypes (modulo response_masks above). Sample shapes differ trivially because each run rolled out independently with seed=1 against different rollout schedules.

### Next steps

- Extend `snapshot_param_grads` to handle DeepSpeed Stage 2 (read partitioned grads from the engine; un-partition for comparison or save partitioned summaries with rank/partition metadata).
- Re-launch grpo_fast step-1 capture; rerun diff for grad statistics.
- Fix `response_masks` dtype mismatch at the capture site (cast to bool in oc, or to int64 in fast — pick whichever matches the trainer's actual usage downstream).

## Bug 6 — `loss_denominator` 60× too big in grpo.py (FIXED)

Step-1 dump diff (rank0) showed `loss_denominator` differs almost exactly by the observed grad-norm ratio:

| | oc | fast | ratio |
|---|---:|---:|---:|
| `loss_denominator` (per accumulation group, all-reduced) | 3,979,577 | 66,718 | 59.6× |
| `reported_grad_norm` (step 1) | 0.0348 | 2.0208 | 58.0× |

Inspecting raw `inputs.response_masks[i]` from the captures:

| | dtype | nunique | max | sum (single mask) |
|---|---|---:|---:|---:|
| oc `response_masks[0]` | int64 | 19 | 34 | 222,681 |
| oc `response_masks[1]` | int64 | 17 | 16 |  74,570 |
| fast `response_masks[0]` | bool | 2 | 1 |   7,182 |
| fast `response_masks[1]` | bool | 2 | 1 |   8,682 |

**oc's `response_masks` are document-ID valued, not binary 0/1.** Each token gets an integer doc-id (0 = pad/non-response, ≥1 = which document inside the pack the token belongs to). `grpo_utils.calculate_token_counts` did `mask[:, 1:].sum()`, which treats the mask as numeric and **sums the doc-id values** instead of counting response tokens. With ~17 docs per pack averaging ~17 tokens each, the sum-of-ids overcounts by roughly the average doc-id (~17×) per pack. After accumulation across two micro-batches and all-reduce across 4 ranks, the inflated denominator was ~60× the true response-token count.

`grpo_fast.py:635` converts `data_BT.response_masks` to `mask.bool()` before the loss path, so fast's denominator is correct. `grpo.py` (oc path) keeps the int64 mask through `calculate_token_counts` and so the doc-id values leak into the denominator.

**Fix (commit `…`)**: in both `grpo_utils.calculate_token_counts` and `grpo_fast.py.calculate_token_counts`, replace `mask[:, 1:].sum()` with `(mask[:, 1:] > 0).sum()` so the function is robust to either binary-bool or doc-id-int64 masks.

Predicted effect: oc's per-step loss and grad_norm rise ~60× to match grpo_fast's; AIME parity should follow.

Validation (beaker `01KQTGZ3DWZ8GNDMQPTH3HF287`, dump at `qwen3_4b_dapo_oc_bug6fix_step1cap_20260504_161526`, diffed against fast baseline `qwen3_4b_dapo_grpofast_step1cap_v5_20260504_151018` in beaker `01KQTHJYHQY23EFDY5YZ4199CV`):

| | Before fix (oc) | After fix (oc) | Fast baseline |
|---|---:|---:|---:|
| `loss_denominator` | 3,979,577 | 66,966 | 66,718 |
| `reported_grad_norm` (step 1) | 0.0348 | 2.2326 | 2.0208 |

After-fix oc still has int64 doc-id-valued `response_masks` (max=18 / 32 in this run, 19 / 15 unique values per pack — confirming the data shape was the input cause), but the new `(mask[:, 1:] > 0).sum()` correctly counts response tokens. Per-rank loss_denominator within 0.4% of fast and grad_norm within 10% (residual variance attributable to independent rollouts on different prompt slices). The 58× grad-norm gap that motivated this whole investigation is fully explained by Bug 6.

Next: rerun a 1000-step training and compare AIME pass@1 to the fast baseline (0.2156) — expect oc to no longer be under-trained at 0.1948.
