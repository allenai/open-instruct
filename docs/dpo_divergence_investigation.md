# DPO Divergence Investigation: dpo.py vs dpo_tune_cache.py

## Problem Statement

`dpo.py` (OLMo-core model) and `dpo_tune_cache.py` (HuggingFace model) produce different training results even when configured identically. Step 1 metrics are close, but divergence compounds rapidly after step 2+.

## Architecture Differences

| Component | `dpo.py` | `dpo_tune_cache.py` |
|-----------|----------|---------------------|
| Model class | `olmo_core.nn.transformer.Transformer` | `transformers.AutoModelForCausalLM` |
| Weight loading | HF → `convert_state_from_hf()` → OLMo-core | Direct `from_pretrained()` |
| Flash Attention API | `flash_attn_varlen_func` (packed) | `flash_attn_func` (batched) |
| Backward call | `loss.backward()` | `accelerator.backward(loss)` |

## Hypotheses & Experiments

### Hypothesis 1: Reference cache batch size mismatch
**Status: CONFIRMED & FIXED**

The reference logprobs cache was generated with `batch_size=4` but training used `batch_size=1`. This caused `ref_logps != policy_logps` at step 1.

**Experiment:** Fixed cache batch size to match training batch size.
**Result:** Step 1 now matches between scripts.

---

### Hypothesis 2: Data ordering mismatch
**Status: RULED OUT**

Both scripts pre-shuffle the dataset with the same seed, then use compatible DataLoader shuffling.

**Experiment:** Added logging to verify data indices match.
**Result:** HFDataLoader logs show POSITIONS; dpo_tune_cache.py logs show ORIGINAL INDICES — they read the same data in the same order.

---

### Hypothesis 3: Forward function strategy mismatch (separate vs concatenated)
**Status: CONFIRMED & FIXED**

`dpo.py` used `--no_concatenated_forward` (separate forward passes), while `dpo_tune_cache.py` used concatenated forward by default.

**Experiment:** Added `--no_concatenated_forward` to `single_gpu_cache.sh`.
**Result:** Both scripts now use the same forward strategy. However, divergence still occurs at step 2+.

---

### Hypothesis 4: Flash Attention kernel API causes gradient differences
**Status: RULED OUT**

We hypothesized that `flash_attn_func` (batched) and `flash_attn_varlen_func` (packed/variable-length) might produce different backward gradients due to different CUDA kernel implementations.

**Experiment:** GPU test directly comparing both APIs with identical Q/K/V tensors:
```python
# flash_attn_func (batched)
out1 = flash_attn.flash_attn_func(q1, k1, v1, causal=True)
out1.sum().backward()

# flash_attn_varlen_func (packed)
out2 = flash_attn.flash_attn_varlen_func(q2, k2, v2, cu, cu, seq_len, seq_len, causal=True)
out2.sum().backward()
```

**Result:** Forward outputs and backward gradients are **exactly identical** (diff=0.0). The flash attention API is NOT a source of divergence.

**Test:** `TestFlashAttnVarlenVsStandardGradients` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 5: Different model implementations produce different autograd graphs
**Status: CONFIRMED**

Even with identical weights loaded into both OLMo-core and HF models, the different code paths (attention layer structure, RoPE application, reshape operations) produce different autograd graphs that yield different gradients.

**Experiment:** GPU test loading identical weights into 2-layer OLMo-core and HF models, running DPO forward+backward:
```python
# Load same weights into both models
hf_state = hf_model.state_dict()
converted = convert_state_from_hf(hf_config, hf_state, ...)
olmo_model.load_state_dict(converted, ...)

# Run forward+backward on same batch
hf_loss.backward()
olmo_loss.backward()

# Compare gradient norms
hf_grad_norm = ...  # Different from olmo_grad_norm
```

**Result:**
- Gradient norms differ between OLMo-core and HF models
- After one optimizer step, logps diverge even more (divergence compounds)

**Test:** `TestOlmoCoreVsHFGradientDivergence` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 7: HF concatenated vs separate mismatch is due to train-mode dropout/RNG
**Status: RULED OUT**

We hypothesized that in train mode, dropout and RNG consumption differences between a single
batch-size-2 forward and two batch-size-1 forwards were causing log-prob divergence.

**Experiment:** In `TestConcatenatedVsSeparateForwardHF`, forced `model.train()`, set all
Dropout modules to `p=0.0`, and snapshot/restore CPU+CUDA RNG state between
`concatenated_forward` and `separate_forward`. Ran on Beaker via
`scripts/test/run_gpu_pytest_dpo_concat.sh` (experiment `01KGRQRQTN134C8XTCZQ73HBXK`,
2026-02-06).

**Result:** Still fails in both cases:
- Same-length: concat `-262.6362` vs sep `-262.6900` (abs diff ≈ `0.054`)
- Different-length: concat `-883.0056` vs sep `-883.7837` (abs diff ≈ `0.778`)

**Conclusion:** Divergence persists without dropout, so the mismatch is likely due to
batch-shape-dependent numerics (batch size 2 vs 1), BF16 kernel nondeterminism, or subtle
masking/attention differences. A deterministic fix likely requires equalizing shapes
or using a single forward path for both.

---

### Hypothesis 6: RoPE computation differs (CPU vs GPU)
**Status: RULED OUT (no effect)**

HF computes `inv_freq` on CPU during `__init__`, OLMo-core computes on GPU. We hypothesized that floating point exponentiation might produce different results.

**Experiment:** GPU test comparing OLMo-core models with and without the RoPE CPU patch, both against HF baseline:
```python
# Build models with same HF weights
unpatched_olmo = build_without_patch()
patched_olmo = build_with_patch()

# Compare gradient norms against HF
```

**Result:**
- HF grad norm: 251.35
- Unpatched OLMo grad norm: 265.74 (diff from HF: 14.39)
- Patched OLMo grad norm: 265.74 (diff from HF: 14.39)
- Unpatched and patched produce **identical** results

**Conclusion:** The RoPE CPU patch has **no measurable effect** on gradient divergence. The ~14 gradient norm difference between OLMo-core and HF comes entirely from other architectural differences (attention implementation, reshaping, etc.).

**Test:** `TestRoPEPatchEffect` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 8: Using HFMatchedOlmo2 in dpo.py to replicate HF computation path
**Status: PARTIAL — step 1 matches, logps differ from step 1**

To eliminate the model implementation difference, we built `HFMatchedOlmo2` (`open_instruct/hf_matched_olmo.py`) — a custom `nn.Module` that replicates HuggingFace's exact computation path (same RMSNorm placement, same RoPE, same `flash_attn_func` call). GPU tests confirmed it produces bit-for-bit identical logits and gradients to `Olmo2ForCausalLM`.

We wired `HFMatchedOlmo2` into `dpo.py` via `--use_hf_matched_model` and ran `scripts/train/debug/dpo/single_gpu.sh`.

**Comparison:** `dpo.py` (HFMatchedOlmo2, run `oztnsjq5`) vs `dpo_tune_cache.py` (run `puxyp9c4`):

| Step | Metric | dpo.py (HF-matched) | dpo_tune_cache.py | Diff |
|------|--------|--------------------:|------------------:|-----:|
| 1 | train/loss | 0.6931471825 | 0.6931471825 | 0.00e+00 |
| 1 | logps_chosen | -48.3957 | -48.1880 | 2.08e-01 |
| 1 | logps_rejected | -57.7561 | -57.6430 | 1.13e-01 |
| 1 | rewards_chosen | 0.0000 | 0.0000 | 0.00e+00 |
| 1 | rewards_rejected | 0.0000 | 0.0000 | 0.00e+00 |
| 2 | train/loss | 0.6915 | 0.6931 | 1.69e-03 |
| 4 | train/loss | 0.6570 | 0.7074 | 5.04e-02 |

Step 1 loss and rewards match exactly (rewards are 0 at step 1 by construction). However, **logps already differ at step 1** (chosen diff=0.21, rejected diff=0.11), which means the forward pass produces different logits even though `HFMatchedOlmo2` is verified identical to HF on GPU.

**Possible remaining causes:**
- The `puxyp9c4` run was from an older commit — the data pipeline or collator may have changed since then
- `dpo_tune_cache.py` uses `accelerate` which may apply different padding or attention masking
- Different tokenizer padding behavior between the two scripts' collators

**Next step:** Launch a fresh `dpo_tune_cache.py` run on the same commit and compare (experiment `01KGT3AHBVJZ9XKPJKDWV3EYCZ` launched, pending results).

### Hypothesis 9: Data pipeline divergence — different inputs reach the model
**Status: RULED OUT**

We compared dpo.py (OLMo-core, experiment `01KGT3XTA4PF3NKPXB3HFE39X7`) vs dpo_tune_cache.py (HF, experiment `01KGT3AHBVJZ9XKPJKDWV3EYCZ`) on the same commit. Step 1, batch 0 (index=2):

| Property | dpo.py (OLMo-core) | dpo_tune_cache.py (HF) | Match? |
|----------|-------------------:|----------------------:|--------|
| Index | `[2]` | `[2]` | YES |
| Chosen logits shape | `[1, 635, 100352]` | `[1, 635, 100352]` | YES |
| Rejected logits shape | `[1, 628, 100352]` | `[1, 628, 100352]` | YES |
| Chosen loss_mask_sums | `[21]` | `[21]` | YES |
| Rejected loss_mask_sums | `[14]` | `[14]` | YES |
| per_token_logps[0,:10] | `[-10.596, -8.594, ...]` | `[-10.596, -8.594, ...]` | YES (bit-for-bit) |
| chosen_logps | **-48.3295** | **-48.1880** | NO (diff=0.14) |
| rejected_logps | **-57.7391** | **-57.6430** | NO (diff=0.10) |

**Conclusion:** The data pipeline is identical — same indices, same shapes, same label masks. The per_token_logps at positions 0-9 are **bit-for-bit identical** between OLMo-core and HF, but the final logps sums differ. Since the 21 labeled tokens are at the END of the 635-token sequence, the divergence accumulates at later sequence positions.

**Key insight:** The matching per_token_logps at early positions confirm the models compute the same function — the divergence is purely numerical precision accumulation over sequence length.

---

### Hypothesis 10: Per-token logps diverge at later sequence positions
**Status: SUPERSEDED by Hypothesis 11**

Added detailed per-token logps logging at sampled positions (every 100 positions + labeled positions + cumulative sums). When comparing OLMo-core (`separate_forward_olmo`) vs HF (`separate_forward`), found that some sequences match perfectly while others diverge. However, this comparison mixes model implementation differences with infrastructure differences.

---

### Hypothesis 11: HF flash_attention_2 wrapper uses different code path than direct flash_attn_func
**Status: RULED OUT**

We hypothesized that `HFMatchedOlmo2` (calling `flash_attn_func` directly) and `Olmo2ForCausalLM` (going through HF's `_upad_input` → `flash_varlen_fn` path) produce different numerical results.

**Experiment:** Modified `HFMatchedOlmo2` to call `transformers._flash_attention_forward` (same path as HF) and pass `attention_mask`. Commit `8ae8afea0`.

**Result:** The fix had **zero effect** — per_token_logps were identical before and after the change. With all-1s attention_mask, `_upad_input` is a no-op (all tokens are kept), so both paths produce the same result. This is consistent with Hypothesis 4.

The divergence observed between `01KGT7DJK16BXJ28Q51ZR1KEW7` (HFMatchedOlmo2) and `01KGT6H4PY5PZWQT0XNW0MH3SP` (Olmo2ForCausalLM) is NOT from the attention code path.

---

### Hypothesis 12: Cross-node non-determinism on H100 GPUs
**Status: RULED OUT**

Same-node test (`01KGTB38W6F15W49XKK5S5SDNV`) ran both HFMatchedOlmo2 and Olmo2ForCausalLM sequentially in the same Beaker job. Results still diverged from position 1 onwards (max_diff=0.2), ruling out cross-node non-determinism.

Within each run, the same model produces bit-for-bit identical results for the same data across multiple forward calls — so flash attention IS deterministic within a process. The divergence is between the two different code paths.

---

### Hypothesis 13: HF's flash_attention_mask converts all-1s mask to None, changing the kernel
**Status: RULED OUT**

HF's `create_causal_mask` → `flash_attention_mask` checks `if attention_mask.all(): attention_mask = None`, converting all-1s masks to `None`. This causes `_flash_attention_forward` to take the `else` branch and call `flash_fn` (regular `flash_attn_func`).

HFMatchedOlmo2 was passing the all-1s mask directly to `_flash_attention_forward`, which took the unpadding branch: `_upad_input` → `flash_varlen_fn` (`flash_attn_varlen_func`).

**Fix attempted:** Pass `attention_mask=None` to `_flash_attention_forward` in HFMatchedOlmo2 (commit `bf4fbc434`), matching HF's code path so both use `flash_attn_func`.

**Result:** Same-node test (`01KGTCMYJXA5T85QVRSV3GBJ28`) showed **no improvement** — per_token_logps still diverge with max_diff=0.2. This is consistent with Hypothesis 4 (flash_attn_func and flash_attn_varlen_func produce identical results). The kernel choice is NOT the cause.

---

### Hypothesis 14: Process-level non-determinism between torchrun invocations
**Status: RULED OUT**

**Test:** Ran `dpo.py --use_hf_matched_model` (HFMatchedOlmo2) **twice** in the same Beaker job (`01KGTEAMHRCRZK2YGNAE7AK1ME`), two separate torchrun invocations with the same model class.

**Result:** All entries match with **max_diff=0.00e+00** — bit-for-bit identical across separate processes. Process-level non-determinism is NOT the cause.

**Conclusion:** There IS a real code difference between HFMatchedOlmo2 and Olmo2ForCausalLM that the GPU unit test did not catch. The divergence starts at position [0+1] (first multi-token attention), suggesting the difference is in either the attention computation, rotary embedding application, or normalization.

**Next step:** Find the exact code difference. The GPU unit test may have used different dtype, shorter sequences, or simpler inputs that masked the divergence.

---

### Hypothesis 15: inv_freq dtype bug — HFMatchedOlmo2 loses precision through `.to(bfloat16)`
**Status: CONFIRMED & FIXED**

`HFMatchedOlmo2`'s `HFMatchedRotaryEmbedding.inv_freq` buffer was cast to bfloat16 by `model.to(dtype=torch.bfloat16)`. HF's `from_pretrained(torch_dtype=bfloat16)` casts parameters but NOT buffers, keeping `inv_freq` in float32. Our `_apply` override now saves inv_freq before any dtype cast and restores it after, and `from_hf_model` copies the exact inv_freq from the HF model.

GPU tests now confirm all 16 layers, 1024 tokens, every component is bit-for-bit identical (max_diff=0.0).

**Commits:** `491e50a01`, `37fea3f1b`, `e32613533`, `cdb1804c4`

---

### Hypothesis 16: dpo.py ignores `max_train_steps`, runs full epochs instead
**Status: CONFIRMED & FIXED**

`dpo.py` set `max_duration=Duration.epochs(num_epochs)` and never used `max_train_steps`. Meanwhile, `dpo_tune_cache.py` checks `completed_steps >= max_train_steps` to stop early.

With `--max_train_steps 5`, dpo.py ran 276 steps (3 full epochs × 92 steps/epoch) while dpo_tune_cache.py ran 5.

**Fix:** Use `Duration.steps(max_train_steps)` when `max_train_steps` is set, otherwise fall back to `Duration.epochs(num_epochs)`.

---

### Hypothesis 17: LR scheduler off-by-one between OLMo-core and HF
**Status: CONFIRMED & FIXED**

The OLMo-core trainer increments `global_step` BEFORE `train_batch()` and `optim_step()`. Inside `optim_step()`, `scheduler.set_lr()` used `global_step` to compute the LR *before* calling `optim.step()`. HF calls `optimizer.step()` first, then `lr_scheduler.step()`. This created an off-by-one error where dpo.py used a lower LR at every step.

With `learning_rate=5e-7`, `warmup_steps=0`, `max_steps=5`, linear decay:

| Step | dpo.py LR (before fix) | dpo_tune_cache.py LR | Diff |
|------|----------------------:|---------------------:|-----:|
| 1 | 4e-7 (`get_lr(5e-7, 1, 5)`) | 5e-7 (initial) | -20% |
| 2 | 3e-7 | 4e-7 | -25% |
| 3 | 2e-7 | 3e-7 | -33% |
| 4 | 1e-7 | 2e-7 | -50% |
| 5 | 0 | 1e-7 | -100% |

**Code trace (before fix):**
```
# OLMo-core trainer loop (trainer.py:1320-1336):
global_step += 1          # step = 1
train_batch(batch)        # forward + backward
optim_step():
    scheduler.set_lr()    # LR = get_lr(5e-7, 1, 5) = 4e-7
    optim.step()          # steps with LR=4e-7 ← WRONG, should be 5e-7

# HF (dpo_tune_cache.py:621-625):
optimizer.step()          # steps with LR=5e-7 (initial)
lr_scheduler.step()       # advances to LR=4e-7 for next step
```

**Fix:** Swapped `optim.step()` and `scheduler.set_lr()` in `DPOTrainModule.optim_step()` so the optimizer steps with the current LR first, then the scheduler advances the LR for the next step (matching HF's ordering).

**Post-fix experiment:** dpo.py (`tugo4ku3`) vs dpo_tune_cache.py (`qyuqlywf`):

| Metric | Step 2 diff (before LR fix) | Step 2 diff (after LR fix) | Improvement |
|--------|---:|---:|---|
| loss | 6.75e-02 | 2.32e-02 | ~3x better |
| logps_chosen | 1.14e+00 | 8.00e-01 | ~1.4x better |
| logps_rejected | 3.91e-01 | 3.04e-01 | ~1.3x better |

Step 1 remains bit-for-bit identical after the fix. The ~3x improvement in loss divergence confirms the LR off-by-one was a significant contributor. The remaining divergence is from gradient differences (Hypothesis 18).

**Commit:** `6b38c2469`

---

### Hypothesis 18: Gradient differences between HFMatchedOlmo2 and Olmo2ForCausalLM
**Status: CONFIRMED**

Even with bit-for-bit identical forward passes (confirmed by Hypothesis 15 fix), the two model implementations produce different gradients due to different autograd graphs. This causes weight updates to differ even with identical LR.

**Evidence from Beaker logs** (post-LR fix runs `tugo4ku3` vs `qyuqlywf`):
- Step 1 gradient norms differ by ~0.078% between dpo.py (HFMatchedOlmo2) and dpo_tune_cache.py (Olmo2ForCausalLM)
- The gradient difference is concentrated in attention parameters
- After one optimizer step, the weight differences compound, causing step 2+ logps to diverge

**Conclusion:** The autograd graph difference between HFMatchedOlmo2 and Olmo2ForCausalLM is the remaining source of training divergence. Despite producing identical forward outputs, the backward passes generate slightly different gradients.

---

### Hypothesis 19: GPU test to isolate model vs training loop gradient differences
**Status: CONFIRMED — gradient difference comes from model implementations, NOT training loop**

To determine whether the gradient difference comes from the model implementations themselves or from differences in the training loop (gradient accumulation, loss scaling, etc.), we added a GPU test (`TestHFMatchedGradientComparison`) that:

1. Loads `allenai/OLMo-2-0425-1B` as both `Olmo2ForCausalLM` and `HFMatchedOlmo2`
2. Creates a realistic DPO batch (~600+ token prefix, 20-25 token responses)
3. Runs `separate_forward` (HF) and `separate_forward_hf_matched` (HFMatched) on the same batch
4. Verifies forward logps are bit-for-bit identical
5. Computes DPO loss with identical fake ref logps
6. Calls `.backward()` on both
7. Compares per-parameter gradient norms

**Result** (experiment `01KH1PBPSP9FT4K9P8X7Y4RBBH`):

| Metric | Value |
|--------|-------|
| Forward logps diff | 0.0 (bit-for-bit identical) |
| Loss diff | 0.0 (bit-for-bit identical) |
| Exact gradient matches | 29/179 (16%) |
| Max relative grad diff | 1.39% (`blocks.7.attention.k_norm.weight`) |
| HF total grad norm | 87.536921 |
| Matched total grad norm | 87.543719 |
| Total grad norm diff | 0.0068 (0.0078% relative) |

**Pattern:** Later layers (13-15) have mostly exact matches; earlier layers accumulate larger differences. This is consistent with a small backward-pass difference at each layer that compounds as gradients propagate from the output layer toward the input.

**Conclusion:** The gradient difference originates in the model implementations themselves, not the training loop. The two models produce identical forward outputs but different backward gradients.

**Test:** `TestHFMatchedGradientComparison` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 20: `@use_kernel_forward_from_hub` decorators replace RMSNorm/SiLU backward in HF model
**Status: OPEN**

HF's `Olmo2RMSNorm` and `SiLUActivation` both have `@use_kernel_forward_from_hub(...)` decorators. If hub kernels are installed on the Beaker GPU machines, these decorators replace the forward method with a custom CUDA kernel that registers its own `torch.autograd.Function`. This would mean:
- Forward outputs are numerically identical (fused kernels compute the same math)
- Backward passes differ (fused kernels use a different autograd Function with different saved tensors)

This fits the observed pattern perfectly: bit-for-bit identical forward, ~0.008% gradient norm difference.

**Against:** If hub kernels produced numerically identical forward outputs via different floating-point operation ordering, it would be remarkable. More likely, fused kernels would show at least some forward difference. The bit-for-bit forward match suggests hub kernels are NOT active (decorator is a no-op).

**Test plan:**
1. Add a GPU test that checks whether hub kernels are active:
   ```python
   # Check if the forward method was replaced by the decorator
   import inspect
   print(type(hf_model.model.layers[0].post_attention_layernorm.forward))
   print(inspect.getsource(hf_model.model.layers[0].post_attention_layernorm.forward))
   ```
2. If active: replace `F.silu` with `SiLUActivation()` and `HFMatchedRMSNorm` with `Olmo2RMSNorm` in HFMatchedOlmo2, then re-run gradient comparison.
3. If not active: rule out this hypothesis.

---

### Hypothesis 21: GPU non-determinism between different nn.Module instances
**Status: CONFIRMED — gradient difference is GPU non-determinism, not a code bug**

Even with identical code and identical weights, two different `nn.Module` instances produce different gradients due to non-deterministic GPU operations (cuBLAS, flash attention backward).

**Experiment** (experiment `01KH1VBJAKGWS6JMT3FNZT3GZ5`):
1. Created two HFMatchedOlmo2 instances from the same HF model (same weights, same code)
2. Ran `separate_forward_hf_matched` + DPO loss + `.backward()` on both with identical input
3. Compared per-parameter gradient norms

**Result:**

| Metric | Identical instances (H21) | HFMatched vs HF (H19) |
|--------|---------------------------|------------------------|
| Exact gradient matches | 12/179 (6.7%) | 29/179 (16%) |
| Max relative grad diff | **1.47%** (`blocks.3.attention_norm.weight`) | 1.39% (`blocks.7.attention.k_norm.weight`) |

The gradient difference between two identical HFMatchedOlmo2 instances (1.47% max) is **larger** than the difference between HFMatchedOlmo2 and Olmo2ForCausalLM (1.39% max). The difference pattern is also the same: later layers have smaller differences, earlier layers accumulate larger ones.

**Conclusion:** The gradient differences observed in H18 and H19 are fully explained by inherent GPU non-determinism. HFMatchedOlmo2 and Olmo2ForCausalLM are functionally equivalent — there is no code bug causing the gradient divergence.

**Test:** `test_h21_gpu_nondeterminism` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 22: Flash attention `deterministic` kwarg differs between code paths
**Status: RULED OUT — kwargs are functionally identical**

### Hypothesis 23: `create_causal_mask` routes HF model through `flash_varlen_fn` (unpadding path)
**Status: RULED OUT — both use same routing**

Combined test for H22+H23 (experiment `01KH1VBJAKGWS6JMT3FNZT3GZ5`): Monkey-patched `_flash_attention_forward` at both import paths to capture all kwargs from both models.

**Result** (32 calls per model, all consistent):

| Kwarg | HF (Olmo2ForCausalLM) | HFMatched | Functionally different? |
|-------|----------------------|-----------|------------------------|
| `attention_mask_is_none` | True | True | No |
| `is_causal` | True | True | No |
| `softmax_scale` | 0.0884 | 0.0884 | No |
| `query_length` | 625 | 625 | No |
| `query_shape` | (1, 625, 16, 128) | (1, 625, 16, 128) | No |
| `deterministic` | NOT_PASSED | NOT_PASSED | No |
| `dropout` | 0.0 | NOT_PASSED (default=0.0) | No |
| `use_top_left_mask` | False | NOT_PASSED (default=False) | No |
| `sliding_window` | None | NOT_PASSED | No |
| `softcap` | None | NOT_PASSED | No |
| `target_dtype` | None | NOT_PASSED | No |
| `extra_kwargs` | [attn_implementation, layer_idx, use_cache] | None | No (metadata only) |

All differences are cosmetic (explicit None/False/0.0 vs default). The critical computation parameters (`is_causal`, `attention_mask`, `softmax_scale`, `deterministic`) are identical. Both models use the same flash attention kernel with the same arguments.

**Test:** `test_h22_h23_flash_kwargs_and_routing` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 24: Comprehensive backward hook debugging
**Status: CONFIRMED — divergence pattern consistent with GPU non-determinism**

Instrumented the backward pass of both models with `register_full_backward_hook` on every layer and sub-module (self_attn, post_attn_norm, mlp, post_ff_norm, final_norm, lm_head).

**Result** (experiment `01KH1VBJAKGWS6JMT3FNZT3GZ5`):

| Component | grad_output diff | grad_input diff |
|-----------|-----------------|-----------------|
| final_norm (call 0) | 0.000000 | 0.000000 |
| final_norm (call 1) | 0.000000 | 0.000000 |
| layer_15 (all sub-modules) | 0.000000 | 0.000000 |
| layer_14 (all sub-modules) | 0.000000-0.000029 | 0.000000-0.000023 |
| layer_13 | 0.000028-0.000414 | 0.000007-0.000219 |
| layer_12 | 0.000015-0.000489 | 0.000016-0.001044 |
| ... (divergence grows) | | |
| layer_0 | 0.000232-0.000740 | 0.000037-0.011129 |

**Key observations:**
1. **Last layer (15)**: Exact match (0.000000) for ALL components — the backward pass starts identically
2. **Layer 14**: Tiny differences appear (max 0.000029) — flash attention backward introduces non-determinism
3. **Earlier layers**: Differences grow monotonically as gradients propagate backward
4. **lm_head grad_input differs** (HF=0.282, Matched=0.593) — this is structural: HFMatched's `lm_head` wraps norm+linear while HF's is just linear

This pattern — exact match at the last layer with growing divergence toward earlier layers — is the signature of non-deterministic flash attention backward, confirming H21's conclusion.

**Test:** `test_h24_backward_hook_divergence` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 25: First-step LR mismatch (initial LR vs warmup start)
**Status: CONFIRMED & FIXED**

The OLMo-core scheduler (`LinearWithWarmup`) is applied AFTER `optimizer.step()` in `optim_step()`. The optimizer is created with `lr=5e-7` (the full learning rate). At the first training step, `optimizer.step()` uses this initial LR before the scheduler has a chance to set the warmup value. In contrast, HF's `get_scheduler()` applies the warmup factor at creation time (step 0 → factor=0 → LR=0).

**Evidence:** Raw WandB data comparison of first 5 steps:

| Step | dpo.py loss | cache loss | dpo.py rewards_chosen | cache rewards_chosen |
|------|-------------|------------|-----------------------|----------------------|
| 1 | 0.6931 (ln2) | 0.6931 (ln2) | 0.000 | 0.000 |
| 2 | 0.6289 | 0.6931 (ln2) | 0.066 | 0.000 |
| 3 | 0.7866 | 0.6981 | -0.066 | -0.006 |

At step 2, dpo.py shows model has been updated (rewards≠0) because step 1 used LR=5e-7. dpo_tune_cache.py still shows rewards=0 because step 1 used LR=0.

Verified empirically:
- dpo.py step 1: optimizer uses LR=5e-7 (initial), then scheduler sets LR=1.85e-8
- dpo_tune_cache.py step 1: optimizer uses LR=0 (scheduler applied at creation), then scheduler advances to LR=1.85e-8

After step 1, both schedules are perfectly aligned, but the one-time divergence from the first step persists as a constant offset throughout training (max_abs_loss_diff=0.14, mean~0.032).

**Fix:** Initialize the optimizer's LR to the scheduler's step-0 value before training:
```python
for group in optim.param_groups:
    group["initial_lr"] = group["lr"]
    group["lr"] = scheduler.get_lr(group["initial_lr"], 0, num_training_steps)
```

---

## Root Cause Summary

**Status: RESOLVED — all divergence sources identified and fixed or explained.**

Four bugs have been identified and fixed:

1. **inv_freq dtype bug** (Hypothesis 15): `HFMatchedOlmo2` cast `inv_freq` to bfloat16 via `model.to()`, while HF kept it in float32. Fixed with `_apply` override. GPU tests confirm bit-for-bit identical outputs.

2. **Step count mismatch** (Hypothesis 16): `dpo.py` ignored `max_train_steps` and ran full epochs. Fixed by using `Duration.steps()`.

3. **LR scheduler off-by-one** (Hypothesis 17): OLMo-core set LR before `optim.step()` using the already-incremented `global_step`, causing dpo.py to use a ~20% lower LR at every step. Fixed by swapping `optim.step()` and `scheduler.set_lr()` ordering. This reduced step 2 loss divergence by ~3x.

4. **First-step LR mismatch** (Hypothesis 25): The optimizer was created with the full LR (5e-7) and the OLMo-core scheduler only set the warmup value AFTER the first `optimizer.step()`. This caused the first training step to use 5e-7 instead of the warmup start (0), creating a one-time divergence that persisted as a constant offset throughout training. Fixed by initializing the optimizer's LR to the scheduler's step-0 value.

**Gradient difference explained by GPU non-determinism (H21, CONFIRMED):** The remaining ~0.008% gradient norm difference between HFMatchedOlmo2 and Olmo2ForCausalLM is NOT a code bug — it is inherent GPU non-determinism. Two identical HFMatchedOlmo2 instances with identical weights produce gradient differences of 1.47% max, which is *larger* than the 1.39% max difference between the two different model classes. The backward hook analysis (H24) confirms the pattern: exact match at the last layer, with non-deterministic flash attention backward introducing small differences that accumulate toward earlier layers.

**Flash attention kwargs confirmed identical (H22+H23, RULED OUT):** Both models call flash attention with the same critical parameters (is_causal=True, attention_mask=None, softmax_scale=0.0884, deterministic=NOT_PASSED). Cosmetic differences (dropout=0.0 vs default, extra metadata kwargs) have no computational effect.

**H20 (hub kernel decorators):** Still OPEN but likely irrelevant given H21's conclusion that the gradient difference is within the noise floor of GPU non-determinism.

**Conclusion:** The two model implementations (HFMatchedOlmo2 and Olmo2ForCausalLM) are functionally equivalent. The training divergence after step 1 came from:
1. First-step LR mismatch: dpo.py used 5e-7 at step 1 vs 0 for dpo_tune_cache.py (now fixed)
2. Inherent GPU non-determinism in flash attention backward (~0.008% per step)
3. These small per-step differences compound across optimizer steps

Previously ruled out (not causes):
- ~~Flash attention kernel API differences~~ (Hypothesis 4 — tested identical)
- ~~Flash attention code path (func vs _upad_input→varlen)~~ (Hypothesis 11 — no effect with all-1s mask)
- ~~Flash attention mask→None conversion~~ (Hypothesis 13 — fix had no effect)
- ~~Flash attention kwargs (deterministic, routing)~~ (Hypothesis 22+23 — all critical params identical)
- ~~RoPE CPU/GPU computation differences~~ (Hypothesis 6 — tested no effect)
- ~~Data ordering~~ (Hypothesis 2)
- ~~Forward function strategy~~ (Hypothesis 3 — now matched)
- ~~Reference cache batch size~~ (Hypothesis 1 — now fixed)
- ~~Data pipeline differences~~ (Hypothesis 9 — inputs verified identical)
- ~~Cross-node non-determinism~~ (Hypothesis 12 — same-node test still diverges)
- ~~Weight tying~~ (OLMo-2-0425-1B uses `tie_word_embeddings=false`)
- ~~SiLU activation function~~ (Both use `nn.functional.silu` — ACT2FN["silu"] resolves to same call)
- ~~Model backward pass code bug~~ (Hypothesis 21 — gradient diff is within GPU non-determinism noise floor)

## Open Questions

1. **Can we make OLMo-core use the exact same attention code path as HF?**
   - Would require significant changes to OLMo-core or a custom attention backend

2. **Can we make HF use OLMo-core's attention?**
   - Could potentially write a custom HF model class that wraps OLMo-core

3. **Should we just use one implementation for both scripts?**
   - Simplest solution: have both scripts use the same model class
   - `dpo.py` could use HF model instead of OLMo-core
   - Or `dpo_tune_cache.py` could use OLMo-core (but loses Accelerate benefits)

4. **Is matching results actually necessary?**
   - If both implementations are mathematically correct, the divergence may be acceptable
   - The goal might shift to "verify both produce good models" rather than "make them identical"

## Relevant Files

- `open_instruct/dpo.py` — OLMo-core DPO training (supports `--use_hf_matched_model`)
- `open_instruct/dpo_tune_cache.py` — HuggingFace/Accelerate DPO training
- `open_instruct/dpo_utils.py` — Forward functions (`separate_forward_olmo`, `separate_forward_hf_matched`, `concatenated_forward`, etc.)
- `open_instruct/hf_matched_olmo.py` — `HFMatchedOlmo2`: custom model replicating HF's exact computation path
- `open_instruct/olmo_core_utils.py` — Model config mapping, RoPE patch, HF conversion
- `open_instruct/test_dpo_utils_gpu.py` — GPU tests for hypothesis verification

## WandB Runs

| Experiment | dpo.py run | dpo_tune_cache.py run |
|------------|------------|----------------------|
| Both using `--no_concatenated_forward` (OLMo-core model) | `ai2-llm/open_instruct_internal/runs/i341yfee` | `ai2-llm/open_instruct_internal/runs/puxyp9c4` |
| `--use_hf_matched_model` (HFMatchedOlmo2) vs dpo_tune_cache | `ai2-llm/open_instruct_internal/runs/oztnsjq5` | `ai2-llm/open_instruct_internal/runs/puxyp9c4` |
| Post LR-fix (Hypothesis 17) | `ai2-llm/open_instruct_internal/runs/tugo4ku3` | `ai2-llm/open_instruct_internal/runs/qyuqlywf` |

OLMo-core vs dpo_tune_cache (all 7 metrics differ):
- `train/loss`: max_abs_diff=1.91e-01, max_rel_diff=2.71e-01
- `train/logps_chosen`: max_abs_diff=2.42e+00, max_rel_diff=7.62e-03
- `train/logps_rejected`: max_abs_diff=2.88e+00, max_rel_diff=6.99e-03

HFMatchedOlmo2 vs dpo_tune_cache (all 7 metrics still differ):
- `train/loss`: max_abs_diff=1.82e-01, max_rel_diff=2.76e-01
- `train/logps_chosen`: max_abs_diff=3.02e+00, max_rel_diff=7.26e-03
- `train/logps_rejected`: max_abs_diff=3.37e+00, max_rel_diff=1.08e-02
- Step 1 loss matches exactly, but logps differ from step 1 (chosen diff=0.21, rejected diff=0.11)
- Note: `puxyp9c4` is from an older commit; a fresh dpo_tune_cache.py run is needed for fair comparison

## Beaker Experiments (Hypothesis 10-12)

| Beaker ID | Script | Model | Node | Key finding |
|-----------|--------|-------|------|-------------|
| `01KGT53VCEC6N5YHT6E708MCBA` | dpo.py (OLMo-core) | OLMo-core Transformer | aus-167 | Chosen logps match HF on same node |
| `01KGT5BKAPWAKXMYWA5SVENJ9J` | dpo_tune_cache.py | Olmo2ForCausalLM | aus-167 | Chosen logps match OLMo-core on same node |
| `01KGT6GY10ZKGBD6Y54H5XHW30` | dpo.py (OLMo-core) | OLMo-core Transformer | aus-167 | 5-step run with input logging |
| `01KGT6H4PY5PZWQT0XNW0MH3SP` | dpo_tune_cache.py | Olmo2ForCausalLM | **aus-216** | 5-step run; logps differ from aus-167 runs |
| `01KGT7DJK16BXJ28Q51ZR1KEW7` | dpo.py --use_hf_matched_model | HFMatchedOlmo2 | aus-167 | Same values as pre-fix; differs from aus-216 |
| `01KGT8D9K5YX10HSFBHGVR7TVF` | dpo.py --use_hf_matched_model | HFMatchedOlmo2 (with _flash_attention_forward) | aus-167 | Zero effect from attention path change |
| `01KGTB38W6F15W49XKK5S5SDNV` | Both (sequential same job) | HFMatchedOlmo2 (with mask) then Olmo2ForCausalLM | same node | Confirms divergence is NOT cross-node |
| `01KGTCMYJXA5T85QVRSV3GBJ28` | Both (sequential same job) | HFMatchedOlmo2 (mask=None fix) then Olmo2ForCausalLM | same node | mask=None fix had no effect, still diverges |
| `01KGTEAMHRCRZK2YGNAE7AK1ME` | dpo.py twice (sequential same job) | HFMatchedOlmo2 twice | same node | **Bit-for-bit identical** (max_diff=0.0) — process non-determinism ruled out |

## Git Commits (investigation branch `finbarr/dpo-match-single-gpu`)

Commits that can be reverted if needed:

| Commit | Description | Revertable? |
|--------|-------------|-------------|
| `56728ba28` | Add detailed per-token logps logging in `_get_batch_logps` | Yes — debug logging |
| `249e8235e` | Add input/mask logging to `separate_forward` and `separate_forward_olmo` | Yes — debug logging |
| `1b24f92d1` | Limit debug DPO scripts to 5 training steps | Yes — debug config |
| `fe4ce1738` | Enable `--use_hf_matched_model` in single_gpu.sh | Yes — debug config |
| `7b9aaea34` | Document Hypothesis 11 (flash attention code path) | Doc only |
| `8ae8afea0` | Use `_flash_attention_forward` in HFMatchedOlmo2 + pass attention_mask | Superseded by next commit |
| `ff087ccb3` | Document Hypothesis 12, add experiment table and git log | Doc only |
| `3fbc73d2c` | Add same_node_comparison.sh for sequential same-node test | Yes — debug script |
| `3f753e14a` | Add input logging to separate_forward_hf_matched | Yes — debug logging |
| `bf4fbc434` | Pass attention_mask=None in HFMatchedOlmo2 (match HF's flash_attn_func path) | Had no effect — not the root cause |

To revert all debug changes: `git revert 8ae8afea0 fe4ce1738 1b24f92d1 249e8235e 56728ba28`
