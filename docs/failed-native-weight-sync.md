# Failed vLLM Native Weight Sync Investigation

**Date**: 2026-04-08 to 2026-04-09
**Branch**: `finbarr/vllm-weight-sync`
**vLLM version**: 0.19.0
**Model**: Qwen/Qwen2.5-7B
**Config**: 4 vLLM engines x TP=2, DeepSpeed stage 3, 2 nodes x 8 GPUs

## Summary

We migrated from a custom `WorkerWrap`-based weight sync (using `torch.distributed.broadcast` + `model_runner.model.load_weights`) to vLLM's native `NCCLWeightTransferEngine` API. After the first weight sync, all vLLM engines produce NaN outputs and crash.

19 experiments progressively narrowed the cause. The NaN is NOT introduced by vLLM's layerwise reload — it exists in DS3 weight shards after `model.step()`, before any gathering or sending. The mystery is that training code is identical between the branch and `origin/main`, and the only differences are vLLM engine configuration on separate GPU processes. DS2 with the same engine config works fine.

## Experiments

### Experiment 1: Unpatched vLLM 0.19.0

**Experiment**: [01KNQB2A4APPPHK19CWNVGHK7F](https://beaker.org/ex/01KNQB2A4APPPHK19CWNVGHK7F)
**W&B run**: [xqt73bw0](https://wandb.ai/ai2-llm/open_instruct_internal/runs/xqt73bw0)

| Time | Event | Status |
|------|-------|--------|
| 20:12:44 | Step 0: initial inference with original weights | OK |
| 20:16:15 | Training step 1 completes (grad_norm: 7.6e9) | OK |
| 20:16:16 | Weight sync completes (0.71s, no NCCL errors) | OK |
| 20:16:16 | `layerwise.py:230` "Failed to load weights" for ALL modules | FAIL |
| 20:16:22 | vLLM inference returns NaN | FAIL |

Every module type reports failure, including both parametric and non-parametric.

### Experiment 2: Patched vLLM 0.19.0 (layerwise.py hotfix + DEBUG logging)

**Experiment**: [01KNSM92NT1WEWVRQF7E24GNGB](https://beaker.org/ex/01KNSM92NT1WEWVRQF7E24GNGB)
**W&B run**: [pv1j3lha](https://wandb.ai/ai2-llm/open_instruct_internal/runs/pv1j3lha)

Applied Dockerfile patch from [vllm-project/vllm#38574](https://github.com/vllm-project/vllm/pull/38574): changed `else:` to `elif info.load_numel_total > 0:` in `layerwise.py:230`. Enabled `VLLM_LOGGING_LEVEL=DEBUG`.

| Time | Event | Status |
|------|-------|--------|
| 17:27:33 | Step 0: initial inference with original weights | OK |
| 17:30:34 | Weight sync completes (0.661s) | OK |
| 17:30:34 | All parametric layers: "Processed" (weights fully received) | OK |
| 17:30:34 | `RotaryEmbedding: Failed to load weights` (on all engines) | FAIL |
| 17:30:42 | vLLM inference returns NaN | FAIL |

All parametric layers load successfully with correct element counts:

```
VocabParallelEmbedding: 272498688 / 272498688 → Processed
QKVParallelLinear: 8259840 / 8259840 → Processed
RowParallelLinear: 6422528 / 6422528 → Processed
MergedColumnParallelLinear: 67895296 / 67895296 → Processed
RowParallelLinear: 33947648 / 33947648 → Processed
RMSNorm: 3584 / 3584 → Processed
ParallelLMHead: 272498688 / 272498688 → Processed
```

Only RotaryEmbedding still fails (its `inv_freq` buffer is counted in `load_numel_total`).

### Experiment 3: Sending buffers alongside parameters

**Experiment**: [01KNSNXB533Y6WW4MAHCJDP0XR](https://beaker.org/ex/01KNSNXB533Y6WW4MAHCJDP0XR)

Sent `model.named_buffers()` (including `inv_freq`) alongside `model.named_parameters()`.

| Time | Event | Status |
|------|-------|--------|
| 17:55:30 | Step 0: initial inference with original weights | OK |
| 17:58:54 | Step 1 inference (before any weight sync) | OK |
| 17:58:58 | Step 2 inference (before any weight sync) | OK |
| 17:59:02 | First weight sync completes (0.659s, grad_norm: 1.41) | OK |
| 17:59:02 | `RotaryEmbedding: Failed to load weights` (still!) | FAIL |
| 17:59:13 | Step 3 inference (after weight sync) → NaN | FAIL |

Sending buffers did not help — `load_weights()` does not handle `inv_freq` (it's computed from config, not a checkpoint weight), so `load_numel` stays 0 for RotaryEmbedding.

Key observations:
- grad_norm was 1.41 (healthy) — training produces valid weights
- Steps 1 and 2 use original weights (inference before sync) and work fine
- NaN only appears for inference AFTER the first weight sync

## What We Ruled Out

### 1. HF → vLLM name mapping: CORRECT

`NCCLWeightTransferEngine.receive_weights()` calls `model.load_weights()` (confirmed in `gpu_worker.py:update_weights`), which goes through `Qwen2Model.load_weights()` with full `stacked_params_mapping`:
- `q_proj` / `k_proj` / `v_proj` → fused `qkv_proj` (shard ids "q", "k", "v")
- `gate_proj` / `up_proj` → fused `gate_up_proj` (shard ids 0, 1)

The layerwise `online_process_loader` wraps individual parameter `weight_loader` methods, so the rename happens BEFORE the wrapper sees the tensor.

### 2. `packed=False` bypassing `load_weights()`: NOT THE CASE

`packed` controls NCCL transport batching. `is_checkpoint_format` (defaults to `True`) controls whether `load_weights()` is called. They are independent fields in `NCCLWeightTransferUpdateInfo`. Our `packed=False` does not affect the loading path.

### 3. NCCL transport: CORRECT

Weight sync completes in ~0.66s with no NCCL errors. All parametric layers show correct element counts matching expected values:
- QKV: 8,259,840 = q_proj (6,422,528) + k_proj bias (256) + v_proj bias (256) + k_proj (917,504) + v_proj (917,504) + q_proj bias (1,792) ✓
- MLP gate_up: 67,895,296 = 2 × 33,947,648 (gate + up per TP shard) ✓

### 4. Cudagraph invalidation: NOT NEEDED

The layerwise reload uses `param.data.copy_()` to write new weights into the original tensor storage (preserving cudagraph-captured memory addresses). This is correct by design — cudagraphs read from the same addresses on replay, so in-place updates are sufficient. No recapture needed.

### 5. Training producing bad weights: NOT THE CASE

Experiment 3 had grad_norm 1.41 (healthy). Steps 1-2 (which use pre-sync weights) produce valid outputs. NaN is exclusively post-sync.

## Remaining Hypothesis

After 4 experiments, we can rule out RotaryEmbedding/inv_freq as the cause (experiment 4 with SKIP_TENSORS didn't help) and rule out bad training weights (experiment 3 with healthy grad_norm 1.41 still NaN'd). The bug is in vLLM's layerwise reload path itself.

Most likely causes:
1. **`_layerwise_process` copy bug**: The `param.data.copy_()` step may not correctly update cudagraph-captured memory for all layers, despite counts showing complete. The copy writes to materialized tensors, but the cudagraph may have captured pointers to the *original* tensor storage that was invalidated during the meta device transition.
2. **Race condition**: The reload may not be fully synchronized before the next inference request replays a cudagraph. `finalize_layerwise_reload()` calls `torch.accelerator.synchronize()`, but this may not be sufficient if the engine core is in a different process.
3. **Meta device transition corruption**: The layerwise reload moves all layers to meta device, then materializes them one-by-one. Even though `param.data.copy_()` writes back into the original storage, the meta→device round-trip may lose tensor metadata (strides, storage offsets) that compiled code depends on.

## How The Old Approach Worked (And Why It Didn't Have This Bug)

The old `WorkerWrap` approach:
1. Trainer broadcasts each parameter via `torch.distributed.broadcast()`
2. Each vLLM worker receives into a fresh tensor
3. Worker calls `self.model_runner.model.load_weights(weights=[(name, weight)])`

This called `load_weights()` directly **without triggering the layerwise reload path**. No meta device transitions, no `_place_kernel_tensors`, no `_layerwise_process`. Weights were applied in-place through the model's own weight loading logic.

The `NCCLWeightTransferEngine` uses the layerwise reload path (`initialize_layerwise_reload` → `finalize_layerwise_reload`), which involves moving layers to meta device, buffering weights, materializing, copying, and restoring kernel tensors. This is a fundamentally different code path with more opportunities for bugs.

## How The New Approach Works

### Engine creation

```python
weight_transfer_config=WeightTransferConfig(backend="nccl")
```

### Weight update

```python
names, dtype_names, shapes = _collect_weight_metadata(model, name_mapper)
refs = [engine.update_weights.remote(names, dtype_names, shapes, packed=False)
        for engine in vllm_engines]

mapped_params = [(name, param.data) for name, param in model.named_parameters()]
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=iter(mapped_params),
    trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group, packed=False)
)
```

### vLLM internal flow (confirmed via source reading)

1. `gpu_worker.update_weights()` calls `initialize_layerwise_reload(model)` — moves all layers to meta, saves original tensors in `info.kernel_tensors`
2. `NCCLWeightTransferEngine.receive_weights()` broadcasts each tensor over NCCL, calls `model.load_weights([(name, weight)])` for each
3. `model.load_weights()` → `stacked_params_mapping` renames HF→vLLM names → calls `param.weight_loader()` per parameter
4. `online_process_loader` (layerwise wrapper) intercepts, buffers weights, tracks `load_numel`
5. When `load_numel >= load_numel_total`: `_layerwise_process()` materializes layer, applies weights, copies into original tensor storage via `param.data.copy_()`, calls `_place_kernel_tensors`
6. For RotaryEmbedding (`load_numel == 0`, `load_numel_total > 0`): skips `_layerwise_process`, directly calls `_place_kernel_tensors` (restores original pre-reload tensors)
7. `finalize_layerwise_reload()` runs, `torch.accelerator.synchronize()`

## Experiment 4: SKIP_TENSORS fix

**Experiment**: [01KNSVC32H28S05D61R7GZ30ZA](https://beaker.org/ex/01KNSVC32H28S05D61R7GZ30ZA)
**W&B run**: [32v8cw1k](https://wandb.ai/ai2-llm/open_instruct_internal/runs/32v8cw1k)

The vLLM team suggested adding `inv_freq` to `SKIP_TENSORS` in `vllm/model_executor/model_loader/reload/meta.py` ([source](https://github.com/vllm-project/vllm/blob/0d310ffbebe588972fb57b84b3ce564c0222ef4e/vllm/model_executor/model_loader/reload/meta.py#L24)). This set controls which tensors are excluded from the layerwise reload's meta device transition and `load_numel_total` tracking. Adding `inv_freq` means:

- RotaryEmbedding's `inv_freq` won't be counted in `load_numel_total`
- `load_numel_total` will be 0 for RotaryEmbedding → no `_place_kernel_tensors` call
- `inv_freq` stays in its original CUDA memory untouched throughout the reload

Applied via Dockerfile `sed` patch on `meta.py`. **Did NOT include the layerwise.py hotfix from experiment 2.**

| Time | Event | Status |
|------|-------|--------|
| 19:30:07 | W&B run starts, Ray cluster connects | OK |
| 19:33:36 | Weight sync thread starts, training step 1 begins | OK |
| 19:33:58 | Training step 1 completes (grad_norm: 50,544,750,592) | BAD |
| 19:33:58 | All parametric layers: "Processed" (weights fully received) | OK |
| 19:34:23 | Non-parametric modules: "Failed to load weights" (SiluAndMul, Qwen2MLP, ModuleList, RotaryEmbedding, ApplyRotaryEmb, LogitsProcessor) | WARN |
| 19:34:24 | vLLM inference returns NaN | FAIL |

**Result**: Same NaN failure. The SKIP_TENSORS fix did not help. Non-parametric module warnings reappeared because the layerwise.py hotfix was not included.

Note: grad_norm was 50.5e9 (catastrophically exploded), but experiment 3 proved that even with healthy grad_norm (1.41), NaN still occurs. The root cause is in the layerwise reload path, not weight quality.

## Summary Across All Experiments

| Exp | layerwise.py hotfix | SKIP_TENSORS inv_freq | grad_norm | Non-param warnings | Result |
|-----|---------------------|-----------------------|-----------|--------------------|--------|
| 1 | No | No | 7.6e9 | ALL modules fail | NaN |
| 2 | Yes | No | — | Only RotaryEmbedding | NaN |
| 3 | Yes (+ sent buffers) | No | 1.41 | Only RotaryEmbedding | NaN |
| 4 | No | Yes | 50.5e9 | All non-parametric | NaN |
| 5 | No | No | — | All non-parametric | NaN |
| 6 | No | No | — | ALL modules fail | NaN |
| 7 | Yes | No | — | Only RotaryEmbedding | NaN |
| 8 | Yes | No | — | Only RotaryEmbedding | NaN |
| 9 | Yes | No | — | Only RotaryEmbedding | NaN |
| 10-13 | N/A | N/A | N/A | N/A | ✅ (minimal repros, no DS) |
| 14 | Yes | No | healthy | N/A | ✅ (DS2 works) |
| 15 | Yes | No | 1.00, 1.41 | N/A | NaN (in gathered tensors before send) |
| 16 | Yes | No | 582M | N/A | NaN (in ds_tensor shards before gather) |
| 17 | Yes | No | 1.00 | N/A | NaN (crash-on-NaN, SP=1) |
| 18 | Yes | No | 9B | N/A | NaN (weight sync timeout, SP=2) |
| 19 | Yes | No | 67B | N/A | NaN (model.step() introduces NaN) |

**Conclusion**: The NaN is NOT caused by vLLM's layerwise reload or weight sync transport. Experiments 10-14 confirmed the weight sync works correctly without DS3. Experiments 15-19 progressively narrowed the cause: NaN exists in `ds_tensor` shards after `model.step()`, before any gathering or sending occurs. The training itself diverges under DS3 on this branch, despite identical training code — the only differences are vLLM engine configuration (`NCCLWeightTransferEngine` vs `WorkerWrap`) on separate GPUs.

## Experiment 5: Parameter name logging

**Experiment**: [01KNSXXVFBJ5T9TKTNF2KB55ZA](https://beaker.org/ex/01KNSXXVFBJ5T9TKTNF2KB55ZA)
**W&B run**: [0u6i2s20](https://wandb.ai/ai2-llm/open_instruct_internal/runs/0u6i2s20)

Added logging to `_collect_weight_metadata` and `broadcast_weights_to_vllm` to compare parameter names sent by our system vs the working vLLM RLHF example. Did NOT include the layerwise.py hotfix.

| Time | Event | Status |
|------|-------|--------|
| 20:14:07 | Job starts | OK |
| 20:18:22 | Weight sync thread starts | OK |
| 20:18:45 | First weight sync: 339 params sent, 0.723s | OK |
| 20:18:45 | Non-parametric modules: "Failed to load weights" | WARN |
| 20:19:09 | Second weight sync: 339 params sent, 0.673s | OK |
| 20:19:10 | Non-parametric modules: "Failed to load weights" | WARN |
| 20:19:11 | vLLM inference returns NaN | FAIL |

Parameter names sent (339 total, all standard HF format):
```
First: model.embed_tokens.weight, model.layers.0.self_attn.q_proj.weight, ...
Last: model.layers.27.mlp.down_proj.weight, model.layers.27.input_layernorm.weight, model.norm.weight, lm_head.weight
```

These match exactly what `AutoModelForCausalLM.named_parameters()` produces. **Parameter names are ruled out as a cause.**

Key differences from the working vLLM RLHF example remain:
- Example uses `enforce_eager=True`, we use compiled mode with cudagraphs
- Example calls `llm.sleep(level=0)` / `llm.wake_up(tags=["scheduling"])` around weight sync
- Example uses `packed=True`, we use `packed=False`

## Experiment 6: pause_generation / resume_generation

**Experiment**: [01KNT1V5B10QNHJQ49FF26CPGG](https://beaker.org/ex/01KNT1V5B10QNHJQ49FF26CPGG)
**W&B run**: [4rfetzwl](https://wandb.ai/ai2-llm/open_instruct_internal/runs/4rfetzwl)

Added `pause_generation(mode="wait")` before weight sync and `resume_generation()` after, to prevent inference during the layerwise reload. Did NOT include the layerwise.py hotfix.

| Time | Event | Status |
|------|-------|--------|
| 21:22:19 | Job starts | OK |
| 21:25:59 | Weight sync thread starts | OK |
| 21:26:44 | Weight sync: 339 params sent | OK |
| 21:26:44 | ALL modules: "Failed to load weights" | FAIL |
| 21:27:23 | vLLM inference returns NaN | FAIL |

**Result**: Same NaN failure. Pausing/resuming the scheduler around weight sync does not fix the issue. The bug is in the layerwise reload path itself, not a race condition with inflight inference.

## Experiment 7: layerwise hotfix + pause/resume

**Experiment**: [01KNT2GPX3E4C86MGX1Y59HP40](https://beaker.org/ex/01KNT2GPX3E4C86MGX1Y59HP40)

Combined layerwise.py hotfix with `pause_generation(mode="wait")` / `resume_generation()`. Weight sync took 19.5s due to draining active requests. Only RotaryEmbedding warnings (hotfix working). Still NaN.

## Experiment 8: layerwise hotfix + sleep/wake_up + enforce_eager

**Experiment**: [01KNT3JK097J23WWCP9WYRM1R6](https://beaker.org/ex/01KNT3JK097J23WWCP9WYRM1R6)

Combined layerwise.py hotfix + `sleep(level=0, mode="keep")` / `wake_up(tags=["scheduling"])` + `enforce_eager=True`. This matches the working vLLM RLHF example as closely as possible. Weight sync back to 0.78s. Only RotaryEmbedding warnings. **Still NaN.**

This definitively rules out cudagraphs — `enforce_eager=True` disables all compilation/cudagraph capture, yet the layerwise reload still produces NaN.

## Experiment 10: Minimal repro — AsyncLLMEngine + TP=1 + packed=False

**Experiment**: [01KNTJS4P5QASX7WMZNEQX6N96](https://beaker.org/ex/01KNTJS4P5QASX7WMZNEQX6N96)

Minimal script based on vLLM's `rlhf_nccl.py` example. Uses `AsyncLLMEngine`, TP=1, `packed=False`, real weights (Qwen2.5-1.5B), single-node, 2 GPUs. No DeepSpeed.

**Result**: ✅ No NaN. Weight sync works correctly. Generation identical before and after sync.

## Experiment 11: Minimal repro — LLM + TP=1 + packed=False

**Experiment**: [01KNTJVYDNH9EGJT9VE33P1NJZ](https://beaker.org/ex/01KNTJVYDNH9EGJT9VE33P1NJZ)

Same as experiment 10 but using `LLM` (offline) instead of `AsyncLLMEngine`.

**Result**: ✅ No NaN. Weight sync works correctly.

## Experiment 12: Minimal repro — LLM + TP=2 + packed=False

**Experiment**: [01KNTPC15NJ2QGA1DWSY9ZMKSR](https://beaker.org/ex/01KNTPC15NJ2QGA1DWSY9ZMKSR)

Same as experiment 11 but with TP=2 (3 GPUs: 1 trainer + 2 for TP). Tests whether tensor parallelism triggers the NaN.

**Result**: ✅ No NaN. TP=2 weight sync works correctly.

## Experiment 13: Minimal repro — LLM + TP=2 + packed=False + send p.data

**Experiment**: [01KNVYHDDA0KJCCWCA22F2Z7DF](https://beaker.org/ex/01KNVYHDDA0KJCCWCA22F2Z7DF)

Same as experiment 12 but sends `p.data` (Tensor) instead of `p` (Parameter), matching how our GRPO code sends weights after DeepSpeed gathering.

**Result**: ✅ No NaN. Sending `.data` vs full Parameter makes no difference.

## Experiment 14: Full GRPO with DeepSpeed stage 2

**Experiment**: [01KNVZA9H7EEPSDAS692DZ06F2](https://beaker.org/ex/01KNVZA9H7EEPSDAS692DZ06F2)
**W&B run**: [3ugl5q5w](https://wandb.ai/ai2-llm/open_instruct_internal/runs/3ugl5q5w)

Full GRPO training run identical to experiments 1-9 but with `--deepspeed_stage 2` instead of 3 (and `--sequence_parallel_size 1` since SP requires DS3). 2 nodes × 8 GPUs, Qwen2.5-7B, 4 vLLM engines × TP=2.

**Result**: ✅ No NaN. Multiple weight syncs complete successfully. Training continues normally.

This definitively isolates the bug: **DeepSpeed stage 3 weight gathering produces weights that cause NaN when passed through vLLM's layerwise reload path.**

## What We've Ruled Out (Updated)

1. HF → vLLM name mapping (experiment 2)
2. `packed=False` bypassing `load_weights()` (code analysis)
3. NCCL transport (experiments 2-5)
4. Cudagraph invalidation (experiment 8 — enforce_eager still NaN)
5. Training producing bad weights (experiment 3)
6. RotaryEmbedding / inv_freq (experiment 4)
7. Parameter names (experiments 5, 9)
8. Scheduler state / pause-resume (experiments 6-7)
9. sleep/wake_up (experiment 8)
10. Cudagraphs / compiled mode (experiment 8 — enforce_eager=True)
11. AsyncLLMEngine vs LLM (experiments 10-11 — both work)
12. TP=2 (experiment 12 — works without DeepSpeed)
13. `packed=False` (experiments 10-13 — all work)
14. Sending `p.data` vs `p` (experiment 13 — no difference)

## Root Cause: DeepSpeed Stage 3

DeepSpeed stage 3 is the trigger. All minimal repros without DeepSpeed work. Full GRPO with DS2 works. Only full GRPO with DS3 produces NaN.

Possible mechanisms:
1. **Tensor properties**: DS3 gathered tensors may have different strides, storage offsets, or contiguity that confuse `load_weights()` or the layerwise reload's `param.data.copy_()`.
2. **Multi-rank gathering**: DS3 uses collective ops to materialize full parameters from shards. The gathered tensor on rank 0 may have unexpected memory layout.
3. **Metadata mismatch**: `_collect_weight_metadata` uses `ds_shape` for shapes, but the actual gathered tensor's properties may differ in subtle ways (e.g., transposed storage).

## Experiment 15: contiguous().clone() + diagnostics

**Experiment**: [01KNW2RSWMANJA315D2WMQYDH5](https://beaker.org/ex/01KNW2RSWMANJA315D2WMQYDH5)

Added `_prepare_params_for_sync()` helper that clones params via `.contiguous().clone()` and logs tensor diagnostics (contiguity, views, shape mismatches, NaN/Inf). Also added vLLM-side weight validation via `scripts/patch_gpu_worker.py`.

| Time | Event | Status |
|------|-------|--------|
| 16:15:42 | Job starts | OK |
| 16:17:47 | Weight sync thread starts | OK |
| 16:18:19 | Training steps complete (grad_norm: 1.00, 1.41) | OK |
| 16:18:20 | **BAD WEIGHTS before send**: ~80+ params have NaN | **KEY FINDING** |
| 16:18:20 | vLLM-side validation also confirms NaN after reload | FAIL |
| 16:18:30 | vLLM inference returns NaN | FAIL |

**Key finding**: The weights are NaN **before being sent to vLLM**. The NaN is in the DS3 gathered tensors, not introduced by vLLM's layerwise reload. No non-contiguous or view warnings — tensor layout is fine.

NaN params are predominantly `k_proj`, `v_proj`, `up_proj`, and `gate_proj` weights across all layers. Training grad_norm is healthy (1.0-1.41, clipped), so the optimizer step itself is fine. The corruption happens during `GatheredParameters` reconstruction from shards.

This definitively rules out vLLM's layerwise reload as the cause. The bug is in DeepSpeed stage 3's `GatheredParameters`.

## Experiment 16: DS3 shard-level NaN validation

**Experiment**: [01KNW35XSQ0J86D3HBGJ4WD6RB](https://beaker.org/ex/01KNW35XSQ0J86D3HBGJ4WD6RB)

Added `ds_tensor` shard validation before `GatheredParameters` to determine whether NaN is in the shards or introduced by the gather.

| Time | Event | Status |
|------|-------|--------|
| 16:22:43 | Job starts | OK |
| 16:24:44 | Weight sync thread starts | OK |
| 16:25:16 | Training step 1 completes (grad_norm: 581,990,848) | BAD |
| 16:25:16 | **NaN in DS3 shards BEFORE gather**: 82 params | **KEY FINDING** |
| 16:25:17 | BAD WEIGHTS before send (after gather): same 82 params | FAIL |

**Key finding**: The NaN is in the `ds_tensor` shards themselves, not introduced by `GatheredParameters`. The training step produced a catastrophic `grad_norm` of 582M, which caused NaN in the optimizer states and weights.

## Updated Root Cause (after experiment 16)

**The NaN is a DS3 training instability, not a weight sync or vLLM bug.** DS3 training with this config produces gradient explosions (grad_norm 582M in exp 16, 67B in exp 19) that cause NaN in optimizer states and weight shards. DS2 with identical hyperparameters (experiment 14) is stable.

However, the training code is byte-for-byte identical between the branch and `origin/main`, where DS3 works. See the "Updated Root Cause Analysis" section after experiment 19 for the current understanding.

## Experiment 17: SP=1, crash-on-NaN checks

**Experiment**: [01KNWN21Q281WD69625YP2V1FF](https://beaker.org/ex/01KNWN21Q281WD69625YP2V1FF)

Changed shard validation from logging to `raise ValueError` so the run crashes immediately on NaN instead of proceeding. Used `--sequence_parallel_size 1` to reduce variables. Same DS3 config otherwise.

| Time | Event | Status |
|------|-------|--------|
| — | Training steps complete (grad_norm: 1.00, clipped) | OK |
| — | NaN in DS3 shards BEFORE gather: 81 params | **CRASH** |
| — | Gradient NaN check did not fire (DS3 doesn't expose `param.grad`) | NOTE |

**Result**: NaN in shards before gather, despite healthy grad_norm (1.00). The gradient NaN check was ineffective because DS3 does not expose `param.grad` on `model.module` parameters — gradients are reduced and partitioned internally by DeepSpeed.

## Experiment 18: SP=2, matching main config

**Experiment**: [01KNWRCSQ6W0HRJG33888E23FQ](https://beaker.org/ex/01KNWRCSQ6W0HRJG33888E23FQ)

Restored `--sequence_parallel_size 2` to match `origin/main`'s `large_test_script.sh` config exactly.

| Time | Event | Status |
|------|-------|--------|
| — | Training steps complete (grad_norm: 9,000,000,000) | BAD |
| — | Weight sync timed out (120s) | FAIL |

**Result**: Catastrophic gradient explosion. Weight sync timed out before crash-on-NaN could trigger.

## Experiment 19: SP=2, pre/post model.step() diagnostics

**Experiment**: [01KNYM0YW1K0YJQW31PBFV7QQW](https://beaker.org/ex/01KNYM0YW1K0YJQW31PBFV7QQW)

Added DS3 shard NaN checks immediately before and after `model.step()` to pinpoint when NaN is introduced.

| Time | Event | Status |
|------|-------|--------|
| — | Loss: 0.03, rewards normal | OK |
| — | DS3 shards clean BEFORE model.step() | OK |
| — | Pre-clip grad norms: empty list (DS3 doesn't expose `param.grad`) | NOTE |
| — | grad_norm (post-clip): 67,270,000,000 | BAD |
| — | **NaN in DS3 shards AFTER model.step()**: 57 params (rank 0), 43 params (other ranks) | **KEY FINDING** |

**Key finding**: `model.step()` is where NaN is introduced. Shards are clean before the step, NaN after. The pre-clip gradient norm check returns an empty list because DS3 partitions gradients internally — `param.grad` is always `None` on `model.module` parameters. The reported `grad_norm` of 67 billion (post-clip, with `max_grad_norm=1.0`) indicates catastrophic gradient values that DeepSpeed's internal clipping could not save.

## Updated Root Cause Analysis

**The NaN is introduced by `model.step()` under DS3.** Experiments 17 and 19 confirmed:
- Shards are clean before `model.step()`
- Shards contain NaN after `model.step()`
- `grad_norm` is catastrophically large (67B even after clipping with `max_grad_norm=1.0`)
- Loss (0.03) and rewards are normal — the forward pass and reward computation are fine

**The central mystery: why does DS3 training diverge only on this branch?**

The training code (forward pass, loss computation, backward pass, optimizer config) is byte-for-byte identical between the branch and `origin/main`. The ONLY differences are in vLLM engine configuration:

1. **Engine creation**: `weight_transfer_config=WeightTransferConfig(backend="nccl")` (branch) vs `worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap"` (main)
2. **Sleep/wake_up calls**: Branch adds `engine.sleep.remote()` before sync and `engine.wake_up.remote()` after
3. **`setup_model_update_group()`**: Branch uses `NCCLWeightTransferEngine.trainer_init()` instead of `vllm_utils.init_process_group()`

These are all vLLM-side operations on separate Ray actor processes running on separate GPUs. They should have no effect on DS3 training, which happens on the trainer GPUs. Yet DS3 training consistently produces catastrophic gradient explosions only on this branch.

**Possible mechanisms (unverified):**
1. **NCCL group interaction**: `NCCLWeightTransferEngine.trainer_init()` creates an NCCL communicator group that may interfere with DS3's internal NCCL groups (used for all-reduce, all-gather). If the groups share resources or the initialization changes NCCL state globally, it could cause silent corruption in DS3's collective operations.
2. **GPU memory pressure**: The `WeightTransferConfig` approach may allocate different NCCL buffers or GPU memory compared to `WorkerWrap`, changing DS3's available memory and triggering different partitioning or offloading behavior.
3. **Timing/ordering**: The `sleep.remote()` call before weight sync may interact with DS3's async operations in unexpected ways, though this seems unlikely since they're on different processes.

**Contradictions to investigate:**
- Experiment 3 (early code) had healthy grad_norm (1.41) but still NaN'd after weight sync. This may represent a *different* bug (vLLM layerwise reload issue) that was subsequently fixed, while the DS3 training instability is a separate, more common issue.
- DS2 with the same `NCCLWeightTransferEngine` setup works fine (experiment 14). This means the NCCL group interaction theory would need to be specific to DS3's collective operations.

## Next Steps

1. ~~Experiments 10-13~~ ✅ Minimal repros all pass. Bug is not in vLLM's core weight sync.
2. ~~Experiment 14~~ ✅ DS2 works. DS3 is the trigger.
3. ~~Add weight validation~~ ✅ Experiment 15 confirmed NaN exists before send, not introduced by vLLM.
4. ~~Try `.contiguous().clone()` before sending~~ ✅ Tensor layout is fine. Problem is NaN values.
5. ~~Check DS3 shards pre-gather~~ ✅ Experiment 16 confirmed shards are NaN before gather.
6. ~~Check pre/post model.step()~~ ✅ Experiment 19 confirmed model.step() introduces NaN.
7. **Test NCCL group interaction**: Run DS3 with `NCCLWeightTransferEngine.trainer_init()` but WITHOUT actually syncing weights. If training still diverges, the NCCL group creation is the cause.
8. **Test without sleep/wake_up**: Remove `sleep.remote()` / `wake_up.remote()` calls to rule them out.
9. **Compare NCCL communicator counts**: Check how many NCCL groups exist on trainer GPUs on branch vs main.
10. **Run DS3 on main with branch's engine config**: Cherry-pick only the engine creation change to main to isolate whether it's the engine config or some other branch change.

## Files Changed

- `open_instruct/grpo_fast.py`: `setup_model_update_group()` uses `NCCLWeightTransferEngine.trainer_init()`
- `open_instruct/vllm_utils.py`: `broadcast_weights_to_vllm()` uses `NCCLWeightTransferEngine.trainer_send_weights()`. `LLMRayActor` uses `init_weight_transfer_engine()` and `update_weights()`
- `open_instruct/vllm_utils_workerwrap.py`: Deleted
- Engine creation uses `weight_transfer_config=WeightTransferConfig(backend="nccl")`
