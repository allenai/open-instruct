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

## Bug 5 (status: needs reverification, probe relaunched)

Original finding (commit `05a966756`, parity probe at `01KQAB41231GR717CXTS9H930Y`): a single-rank parity probe loaded identical `Qwen/Qwen3-4B-Base` weights into HF Qwen3 and OLMo-core's `Transformer`, ran the same input, and captured per-block hidden states. Result on the then-current olmo-core pin:

- single-doc, no packing: blocks 0–5 within bf16 noise (≤5e-1 max), **block_6 jumps to max ≈288**, blocks 7–35 stay near 288.
- Net logprob max\|Δ\| 3.1e-2, mean 7.0e-3, argmax-agree 100%.
- Suspected cause at the time: q_norm / k_norm placement, RoPE θ=1e6 mismatch, or RMSNorm precision in OLMo-core's Qwen3 port.

Subsequent commits bumped `ai2-olmo-core` rev several times in pursuit of a fix, ending at the current pin `61091dba9c2761e4ba3294c916cee33739f7ff1a` (commit `37ac6e68a`, "single-doc parity verification"). The probe scripts were deleted in `64e36aabd` ("cleaned up PR") so we have no in-repo evidence the fix held.

I've restored the latest probe (`scripts/diagnostics/olmo_core_hf_parity.py`, originally added at `939a1f631`) and a launch script. **A Beaker run against the current pin is in progress.** Acceptance criterion: per-block max\|Δ\| stays in bf16 noise (~1e-2) all the way through `block_35`, and final logprob max\|Δ\| ≪ 1e-2.

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

1. **Bug 5 still live.** If OLMo-core's Qwen3 forward still diverges from HF at block_6+, the resulting gradients have a fundamentally different shape — the report-side norm and the actual update direction can both be off by large factors even when the per-token loss looks similar. This is the next thing to verify (probe launched).
2. **DTensor norm aggregation under HSDP.** `get_total_norm` on `_NormPartial` DTensors with subsequent `.full_tensor()` should sum `p^norm` across shards before taking the root. A bug here would systematically under-report by ~√shard_degree (i.e. ~2×), not ~50× — but worth ruling out by computing the norm manually after `loss.backward()`.
3. **Activation checkpointing (`activation_memory_budget=0.5`).** Selective AC shouldn't change gradients, but this is the only training-config delta beyond trainer choice.
4. **Optimizer-state lifecycle.** `super().optim_step()` was wired in `008dc95` as the fix for Bug 3. Worth re-confirming the optimizer actually advances every accumulation boundary in a current run (e.g. by logging param hash before/after on rank 0 for a few steps).

## What we know is fine

- `lr` matches across runs at every checkpoint.
- `debug/vllm_vs_local_logprob_diff_mean ≈ 6e-3` in both runs — the trainer's forward agrees with vLLM to the same degree in both trainers, so the *trainer↔vLLM* logprob path isn't the culprit (this metric does **not** test trainer-vs-trainer agreement, only trainer-vs-vllm).
- `val/ratio` and `val/tis_clipfrac` are not diagnostic with `num_mini_batches=1, num_epochs=1` (old logprob = new logprob by construction).
- Bug 5 update from earlier revisions: HF's Qwen3 with `attention_mask=None + position_ids` does **not** build an intra-doc mask — `position_ids` only affect RoPE. So grpo_fast.py's training-time HF forward also has cross-doc attention bleed, which is why the original packed-vs-solo probe didn't reveal a clean structural difference. The dominant issue was always the single-doc block_6 divergence.

## Recommended next moves

1. **Run the parity probe against pin `61091dba`.** In progress. If single-doc per-layer max stays in bf16 noise through `block_35`, Bug 5 is confirmed fixed and we move on to (2). If `block_6` is still ~288, the upstream olmo-core fix didn't actually land or the pin is wrong.
2. **Bisect the grad_norm gap.** Run grpo.py with `fsdp_shard_degree=1, fsdp_num_replicas=4` (DDP-equivalent, no FSDP sharding). If grad_norm jumps back to ~1.0, the issue is FSDP-specific (DTensor norm aggregation, reduce-scatter semantics). If it stays at ~0.02, the issue lives in the loss/forward path independent of FSDP.
3. **Manual grad-norm check.** Add a single-rank instrumented variant that, after `loss.backward()` and before clip, computes `sqrt(sum(p.grad.detach().to_local().pow(2).sum() for p in params))` (DTensor-aware) and prints alongside the clip-reported value. Disagreement would localize the bug to the clip path.
4. **Confirm optimizer is stepping.** Log `id(p) -> p.detach().cpu().abs().mean()` for a fixed parameter before and after a known step on rank 0, in both trainers. Rules out a subtler regression of Bug 3.

## Probe + scripts

- `scripts/diagnostics/olmo_core_hf_parity.py` — single-rank HF↔OLMo-core forward + per-layer hidden-state diff probe.
- `scripts/diagnostics/launch_olmo_core_hf_parity.sh` — Beaker single-GPU launch wrapper. Run via `./scripts/train/build_image_and_launch.sh scripts/diagnostics/launch_olmo_core_hf_parity.sh`.
