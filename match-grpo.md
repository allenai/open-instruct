# Matching grpo.py to grpo_fast.py — Investigation Summary

Goal: get `grpo.py` (OLMo-core trainer) to learn equivalently to `grpo_fast.py` (HF trainer) on the Qwen3-4B-Base DAPO-Math setup.

## Reference runs

| | run | wandb |
|---|---|---|
| Broken grpo (1000 steps) | beaker `01KQ8DSPZSR4BS9Q0JJSRGDWVC` | `ai2-llm/open_instruct_internal/9a371nw1` |
| Reference grpo_fast (1000 steps) | beaker `01KPTSPMH4Q2SBQMKGZ6YA145Q` | `ai2-llm/open_instruct_internal/parozgke` |
| Fixed grpo (≥109 steps so far) | beaker `01KQ8Y1795ZYXKGTP95EDE01FG` | `ai2-llm/open_instruct_internal/jp4rp68f` |

Script under test: `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh`.

## Initial framing trap

The broken grpo run finished 1000 steps in 3h42m vs grpo_fast's 14h22m — looked like a 3.9× speedup. It wasn't:

- `val/sequence_lengths` median 1035 (broken) vs 4018 (ref). Same model, same prompts, same response cap.
- `val/actor_tokens_per_second` was actually **slower** in broken run (2254 vs 4609).
- `objective/math_reward` was flat at 2.7 for the entire broken run; reference grew 4.3 → 5.2.

The "speedup" was a non-learning policy generating short, mostly-wrong responses.

## Bug 1 — script not consuming the built image

`scripts/train/qwen/qwen3_4b_dapo_math_oc.sh` defaulted to `BEAKER_IMAGE=nathanl/open_instruct_auto` and forwarded `"$@"` to `grpo.py`. `build_image_and_launch.sh` passes the freshly-built image as `$1`, so the image arg leaked into `grpo.py`'s argparse as a stray positional, and the run used a stale auto image without `--eval_top_p`.

Fix: consume `$1` as the image, like sibling scripts do:
```bash
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true
```

## Bug 2 — `inflight_updates` kwarg passed to a callback that doesn't accept it

`grpo_olmo_core_actor.py:286` constructed `VLLMWeightSyncCallback(..., inflight_updates=...)`. The `VLLMWeightSyncCallback` dataclass on this branch had no such field — a partial cherry-pick of commit `be4c4eb32` (which is fully on `finbarr/match-grpo`).

Fix: drop the kwarg from the actor call. The callback already syncs synchronously by construction.

## Bug 3 — optimizer never steps

This is the cause of the flat reward / flat sequence length in the broken run.

`olmo_core_train_modules.py:332-340` overrides `optim_step` and `zero_grads` to no-ops:

```python
def optim_step(self) -> None:
    # No-op: GRPO steps the optimizer internally inside train_batch per mini-batch.
    pass

def zero_grads(self) -> None:
    pass
```

The override exists because OLMo-core's outer trainer loop (`trainer.py:1447-1453`) unconditionally calls `optim_step` + `zero_grads` after `train_batch`. Without the override the outer loop would step a second time and re-record `optim/*` metrics, hanging on the gloo `all_gather_object` consistency check.

But `train_batch` itself called `self.optim_step()` (lines 501-509) — which resolves to the override. Net effect: `loss.backward()` runs every microbatch, gradients accumulate, but `optim.step()` is **never** called for the entire run. Parameters stay frozen at init.

This perfectly explains:
- `val/ratio == 1.0` exactly (variance ~1e-16) — but see "False alarm" below.
- `policy/clipfrac_avg == 0` — no ratio deviation to clip.
- Flat reward, flat sequence length.
- `lr` recorded as decaying because the metric pulled `scheduler.get_lr(...)` for display only; the actual `param_group["lr"]` was never written.

Fix: call the parent's implementation explicitly inside `train_batch`:

```python
if local_step % accumulation_steps == 0:
    if not dry_run:
        super().optim_step()
    super().zero_grads()
```

`self.optim_step()` would still hit the override; `super().optim_step()` bypasses it for the inner call only, leaving the outer-loop call as the no-op the override is meant to provide.

## Bug 4 — `--lr_scheduler_type constant` silently routed to linear-decay-to-zero

`grpo_olmo_core_actor.py:142-145`:

```python
if self.grpo_config.lr_scheduler_type == "cosine":
    scheduler = CosWithWarmup(warmup_steps=warmup_steps)
else:
    scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)
```

Anything that wasn't literally `"cosine"` — including the `"constant"` the script passed — became `LinearWithWarmup(alpha_f=0.0)`, decaying lr from the configured value to **0** over `num_scheduler_steps`. The observed lr trajectory `9e-7 → 5e-7 → 1e-7 → 0` was pure scheduler decay being *displayed* (the optimizer wasn't actually stepping, see Bug 3, so it was cosmetic — but it would have bitten as soon as Bug 3 was fixed).

Fix: route explicitly, raise on unknown:
```python
if self.grpo_config.lr_scheduler_type == "cosine":
    scheduler = CosWithWarmup(warmup_steps=warmup_steps)
elif self.grpo_config.lr_scheduler_type == "constant":
    scheduler = ConstantWithWarmup(warmup_steps=warmup_steps)
elif self.grpo_config.lr_scheduler_type == "linear":
    scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)
else:
    raise ValueError(...)
```

## False alarm — `val/ratio == 1.0` is not diagnostic

I initially flagged this as evidence the optimizer wasn't stepping. The reference run also has `val/ratio == 1.0` with variance ~1e-16. Cause: `grpo_utils.py:330-336` with `num_mini_batches=1`, `num_epochs=1`, and default `use_vllm_logprobs=False`:

```python
if epoch_idx == 0 and not use_vllm_logprobs:
    old_logprobs_cache[sample_idx] = new_logprobs.detach()
```

So `log_ratio = new - new.detach() = 0` and `ratio = 1` by construction. The metric only carries information with multiple PPO inner steps. The real diagnostic was the flat reward + flat sequence length, not the ratio.

## Remaining gap — TIS clipping is ~1300× higher in grpo than grpo_fast

After Bugs 1-4 fixed, the run learns. At step 109:

| metric | new (fixed) | ref (grpo_fast) |
|---|---:|---:|
| `lr` | 1e-6 | 1e-6 |
| `objective/math_reward` | 3.50 | 3.88 |
| `val/sequence_lengths` | 1062 | 1899 |
| `val/sequence_lengths_solved` | 853 | 1442 |
| `val/stop_rate` | 0.992 | 0.957 |
| **`val/tis_clipfrac`** | **0.0046** | **3.5e-6** |

In the first ~36 steps everything except `tis_clipfrac` is identical (math_reward 3.06 vs 3.12, seq_lengths 1040 vs 1050). `tis_clipfrac` is ~1300× elevated **from the very first measurement**. This is structural, not a function of training divergence — it's there with identical weights at step 1.

`val/tis_clipfrac` measures the fraction of response tokens where `|log(p_trainer / p_vllm)| > log(cap)`. The trainer recomputes logprobs on rollout tokens, and they disagree with vLLM far more in grpo than in grpo_fast.

Most likely cause: **OLMo-core forward pass ≠ HF/vLLM forward pass on the same Qwen3 weights.** grpo_fast.py uses HF's Qwen3 implementation for both training and old-logprob recompute, so it agrees with vLLM by construction. grpo.py runs an OLMo-core port — if RoPE, RMSNorm precision, fused ops, or attention masking differ, logprobs will systematically diverge even with bit-identical state dicts.

Other candidates worth ruling out:
- bf16 vs fp32 logit precision in the logprob path
- packing/position-id differences when FSDP2 micro-batches don't match vLLM's request batching
- temperature application path mismatch (both nominally 1.0; verify the scaling site is the same)

## Bug 5 — OLMo-core forward silently ignores `attention_mask` and `position_ids`

The TIS gap is structural and explained by a packed-sequence attention bug.

`data_loader.py` packs multiple prompt+response pairs into each training sample. Within a packed sample, `position_ids` restart at 0 per document, and `attention_masks` encode doc boundaries.

- `grpo_fast.py:692` (HF path) passes `attention_mask=None` so HF Qwen3 constructs the correct intra-document attention mask from `position_ids`. This matches vLLM's per-request attention.
- `olmo_core_train_modules.py:431-439` passes both `attention_masks[i]` and `position_ids[i]` into `forward_for_logprobs`, which forwards them as kwargs to `model(...)`.
- **OLMo-core's `Transformer.forward` accepts neither argument.** Inspecting the source: zero references to `attention_mask` or `position_ids`. Both are silently absorbed by `**kwargs` and discarded. OLMo-core derives intra-document attention from `doc_lens`/`max_doc_lens`/`cu_doc_lens`; without those, attention runs across the entire packed sequence, crossing document boundaries.

Net effect on the trainer-side logprob recompute:
- vLLM generated each response with per-request attention (no cross-doc bleed).
- grpo_fast (HF) reproduces that with `position_ids`-derived intra-doc masks.
- grpo.py (OLMo-core) reproduces it with **whole-packed-sequence attention** — every token sees every prior token regardless of document — so logprobs systematically disagree with vLLM.

Hence `tis_clipfrac` is ~1300× elevated from the very first measurement, with identical weights, exactly as observed.

Fix sketch: derive `doc_lens` (or directly `cu_doc_lens` + `max_doc_len`) from `position_ids` at each forward and pass them through `forward_for_logprobs` into `model(...)`. Boundaries are positions where `position_ids` resets to 0.

## Bug 5 update — parity probe results (Beaker `01KQAB41231GR717CXTS9H930Y`)

Single-rank probe loads identical Qwen3-4B-Base weights into HF Qwen3 and OLMo-core's `Transformer` (via `load_hf_model`), runs the same input, captures hidden state after each block via forward hooks.

**Tied lm_head sanity:** `same_storage=False`, but `max|w_emb − w_out|=0` — load path materialized two equal copies, OK.

| | logprob max\|Δ\| | logprob mean\|Δ\| | argmax-agree |
|---|---:|---:|---:|
| single-doc, OLMo BROKEN (no doc_lens) | 3.1e-2 | 7.0e-3 | 100.0% |
| packed-2-doc, OLMo BROKEN | 9.00 | 1.86 | 72.7% |
| packed-2-doc, OLMo FIXED (doc_lens passed) | 7.25 | 1.13 | 68.2% |

**Per-layer single-doc (no packing → rules out doc-mask bugs):**
- `embed`: 0 (weights load identically, including tied lm_head)
- `block_0`: max 0.031 — bf16 noise
- `block_1..5`: max grows 0.5 → 0.5, mean 1e-3 → 3e-3
- **`block_6`: max jumps to 288**, mean still 0.027 — single outlier-channel diverges
- `block_7..35`: max stays ≈288, mean grows 0.029 → 0.235
- `final_norm`: max 3.5, mean 0.028

**Two findings that revise Bug 5:**

1. **Single-doc still diverges.** With no packing and `position_ids = [0..N-1]`, the dropped kwargs are inert — yet OLMo and HF disagree starting at `block_6`. There's a real arch / weight-conversion mismatch beyond the silent-kwarg-drop. This is the dominant cause of the trainer↔vLLM TIS gap, not the packed attention bug.

2. **HF Qwen3 with `attention_mask=None + position_ids` does NOT build an intra-doc mask.** If it did, OLMo FIXED (intra-doc via `doc_lens`) should match HF closer than OLMo BROKEN (cross-doc). Instead BROKEN is closer (mean Δ 1.86 vs FIXED's 1.13 in mean is misleading because of the block_6 outlier — argmax-agree is 73% BROKEN vs 68% FIXED, and max is smaller for BROKEN). HF runs plain causal across the whole packed sequence; `position_ids` only affect RoPE. **`grpo_fast.py:690` comment claiming "HF constructs the correct 3D intra-document mask from position_ids" is incorrect.**

The implication: grpo_fast.py's training-time HF forward also has cross-doc bleed during training and only agrees with vLLM by accident or because most packed boundaries are far enough apart that bleed is small. The 1300× TIS gap is driven by the block_6+ arch divergence in OLMo-core, not packing.

**Where to look next at block_6:**
- q_norm / k_norm (per-head RMSNorm new in Qwen3): present in HF, declared in OLMo-core qwen3_4B config (`qk_norm=True`, `use_head_qk_norm=True`). Verify weights load and that placement/scope (per-head vs per-channel) matches.
- RoPE θ=1e6 — check OLMo-core's RoPE matches HF's (fp32 cos/sin, frequency layout, half-rotate convention).
- RMSNorm precision — HF Qwen3 casts to fp32 inside RMSNorm, then back; check OLMo-core does the same.

## Recommended next moves

1. **Diagnostic: raise `--truncated_importance_sampling_ratio_cap` to 5.0** and confirm reward/length catch up to reference. If they do, TIS clipping is the dampener.
2. **Trainer–vLLM logprob parity check.** Single-rank script: load identical Qwen3 weights into OLMo-core's `TransformerTrainModule` and into vLLM, feed the same prompt+response, compare per-token logprobs. The expected delta is < bf16 noise; anything systematic points at a specific layer.
3. **Add the missing `time/training`, `time/total`, `time/weight_sync` metrics to grpo.py** so future apples-to-apples diffs against grpo_fast don't require manual reconstruction.
4. **Fix the `else: LinearWithWarmup` fallthrough pattern wherever else it appears.** Silent substitution of the user's flag is hostile.

## Commits on this branch

- `833f8dc` Fix qwen3_4b_dapo_math_oc.sh to accept image as $1 from build_image_and_launch.sh
- `0f04b02` Stop passing inflight_updates to VLLMWeightSyncCallback; field doesn't exist on this branch
- `008dc95` Fix GRPO no-op optim_step and constant lr scheduler routing
