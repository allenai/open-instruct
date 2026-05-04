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

## Bug 5 (status: single-doc FIXED, packed/doc_lens path BROKEN)

Original finding (commit `05a966756`): single-doc forward through OLMo-core's Qwen3 port diverged from HF starting at `block_6` (max ≈288), causing the trainer↔vLLM TIS gap.

Re-ran the parity probe (`scripts/diagnostics/olmo_core_hf_parity.py`, restored from `939a1f631`) against the current pin `61091dba` on Beaker `01KQSYMVPRBNTG22WQXY4Y6TVM` (single GPU, ~1 min wallclock).

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

### Packed-2-doc — BROKEN, and the FIXED path is the worse one

The probe runs OLMo-core two ways on the same packed-2-doc input:
- **BROKEN**: `model(input_ids, attention_mask=None, position_ids=position_ids)`. OLMo-core silently drops both kwargs, so attention is plain causal across the whole packed sequence (cross-doc bleed).
- **FIXED**: `model(input_ids, doc_lens=doc_lens, max_doc_lens=max_doc_lens)`. Should give intra-document attention.

Result on current pin:

| | BROKEN (no doc_lens) | FIXED (with doc_lens) |
|---|---:|---:|
| `block_0` max\|Δ\| | 1.05 | 5.69 |
| `block_5` max\|Δ\| | 4.13 | 50.8 |
| **`block_6` max\|Δ\|** | 7.44 | **5405** |
| `block_35` max\|Δ\| | 303 | 798 |
| logprob max\|Δ\| | 9.06 | 7.22 |
| argmax-agree | 72.7% | 72.7% |

**Passing `doc_lens` makes things ~700× worse at block_6**, and the divergence stays in the thousands all the way to block_33. Either the doc_lens code path in olmo-core is buggy, or `forward(..., doc_lens=...)` no longer exists on the current pin and the kwarg is being routed somewhere harmful (silently mutating activations rather than just being dropped).

The training-time forward in `olmo_core_train_modules.py` passes `doc_lens` (commit `8ca1313a6 Pass doc_lens to OLMo-core forward in GRPO logprob recompute`). **So the trainer is currently running the broken-FIXED path on every step.** This is consistent with grpo.py training reward looking superficially OK (the gradient still has signal) while transferring poorly to AIME — gradients computed against logits that are 5000× off from "true" Qwen3 are still a learning signal, just a wrong one.

### Caveats on the packed test

- Argmax-agree being 72.7% in **both** BROKEN and FIXED suggests the cross-doc-bleed BROKEN path has been the de-facto "good enough" path — vLLM/HF also have cross-doc effects for this input shape (per the earlier note: HF's `attention_mask=None + position_ids` does NOT build an intra-doc mask; `position_ids` only affects RoPE).
- The probe input is `"The quick brown fox jumps over the lazy dog." + "In a hole in the ground there lived a hobbit."` (10 + 12 tokens). Real training packs are much longer with more docs; the failure mode there will be different but the doc_lens path is shared.

### Next moves on this

1. **Read olmo-core upstream commit `61091dba` Transformer.forward signature.** If `doc_lens` was renamed/removed, our `forward_for_logprobs` wrapper is passing a kwarg the model now silently mishandles. If still present, reproduce minimum failure inside olmo-core's tests and file upstream.
2. **Try BROKEN-path training**, i.e. drop `doc_lens` from `olmo_core_train_modules.py` forward calls. If grad_norm jumps from 0.02 → ~1.0 and AIME catches up, the doc_lens regression is the dominant cause of all the symptoms.
3. **Trainer-vLLM logprob diff is small (~6e-3) in BOTH runs** despite the parity probe showing 5400-magnitude divergence with doc_lens. That means either (a) actual training packs trigger this less than the toy 2-doc test does, or (b) the way we recompute logprobs in the trainer cancels much of the error. Worth understanding which before committing to (2) — see investigation note below.

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

1. **doc_lens path in OLMo-core is broken (most likely cause).** Parity probe (Bug 5 section above) shows `model(..., doc_lens=...)` produces hidden-state differences of ~5400 at block_6 vs HF, while the kwarg-dropping path gives ~7. The trainer uses the doc_lens path. Even if the trainer↔vLLM logprob diff stays small (because vLLM is also wrong in a similar way, or because most packs are dominated by single docs), the gradients are computed against ~thousands-magnitude-wrong intermediate activations — the resulting grad shape and norm are not the gradient we'd get from a "true" Qwen3 forward.
2. **DTensor norm aggregation under HSDP.** `get_total_norm` on `_NormPartial` DTensors with subsequent `.full_tensor()` should sum `p^norm` across shards before taking the root. A bug here would systematically under-report by ~√shard_degree (i.e. ~2×), not ~50× — but worth ruling out by computing the norm manually after `loss.backward()`.
3. **Activation checkpointing (`activation_memory_budget=0.5`).** Selective AC shouldn't change gradients, but this is the only training-config delta beyond trainer choice.
4. **Optimizer-state lifecycle.** `super().optim_step()` was wired in `008dc95` as the fix for Bug 3. Worth re-confirming the optimizer actually advances every accumulation boundary in a current run (e.g. by logging param hash before/after on rank 0 for a few steps).

## What we know is fine

- `lr` matches across runs at every checkpoint.
- `debug/vllm_vs_local_logprob_diff_mean ≈ 6e-3` in both runs — the trainer's forward agrees with vLLM to the same degree in both trainers, so the *trainer↔vLLM* logprob path isn't the culprit (this metric does **not** test trainer-vs-trainer agreement, only trainer-vs-vllm).
- `val/ratio` and `val/tis_clipfrac` are not diagnostic with `num_mini_batches=1, num_epochs=1` (old logprob = new logprob by construction).
- Bug 5 update from earlier revisions: HF's Qwen3 with `attention_mask=None + position_ids` does **not** build an intra-doc mask — `position_ids` only affect RoPE. So grpo_fast.py's training-time HF forward also has cross-doc attention bleed, which is why the original packed-vs-solo probe didn't reveal a clean structural difference. The dominant issue was always the single-doc block_6 divergence.

## Recommended next moves

In rough priority order:

1. **Investigate the doc_lens path.** Read `Transformer.forward` at olmo-core pin `61091dba` and confirm whether `doc_lens` / `max_doc_lens` are still accepted kwargs and whether they're routed to the attention kernel correctly. The parity probe shows passing them produces ~5400 hidden-state divergence — either the API changed and our wrapper is wrong, or there's an upstream regression.
2. **Try training with doc_lens disabled.** Drop the `doc_lens=...` kwarg from `olmo_core_train_modules.py:forward_for_logprobs` calls and rerun a short grpo.py. If grad_norm jumps to ~1.0 and AIME catches up to the grpo_fast baseline, this is the dominant cause.
3. **Bisect the grad_norm gap independently.** If (2) doesn't close it, run grpo.py with `fsdp_shard_degree=1, fsdp_num_replicas=4` (DDP-equivalent). If grad_norm jumps back to ~1.0 there, the residual is FSDP-specific (DTensor norm aggregation, reduce-scatter semantics). If it stays at ~0.02, the issue is in the loss path.
4. **Manual grad-norm check.** Single-rank instrumented variant that, after `loss.backward()` and before clip, computes `sqrt(sum(p.grad.detach().to_local().pow(2).sum() for p in params))` (DTensor-aware) and prints alongside the clip-reported value. Disagreement localizes the bug to the clip path.
5. **Confirm optimizer is stepping.** Log a fixed parameter's `abs().mean()` before and after a known step in both trainers. Rules out a subtler regression of Bug 3.

## Probe + scripts

- `scripts/diagnostics/olmo_core_hf_parity.py` — single-rank HF↔OLMo-core forward + per-layer hidden-state diff probe.
- `scripts/diagnostics/launch_olmo_core_hf_parity.sh` — Beaker single-GPU launch wrapper. Run via `./scripts/train/build_image_and_launch.sh scripts/diagnostics/launch_olmo_core_hf_parity.sh`.
