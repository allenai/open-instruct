# Failed vLLM Native Weight Sync Investigation

**Date**: 2026-04-08 to 2026-04-09
**Branch**: `finbarr/vllm-weight-sync`
**vLLM version**: 0.19.0
**Model**: Qwen/Qwen2.5-7B
**Config**: 4 vLLM engines x TP=2, DeepSpeed stage 3, 2 nodes x 8 GPUs

## Summary

We migrated from a custom `WorkerWrap`-based weight sync (using `torch.distributed.broadcast` + `model_runner.model.load_weights`) to vLLM's native `NCCLWeightTransferEngine` API. After the first weight sync, all vLLM engines produce NaN outputs and crash.

Three experiments and deep code analysis confirmed that the issue is inside vLLM's layerwise reload path. Our integration code (name mapping, transport, metadata) is correct.

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

Despite all parametric layers showing "Processed" with correct counts, inference after weight sync produces NaN. The only visible anomaly is `_place_kernel_tensors` being called on RotaryEmbedding. However, this should theoretically be safe — it restores the original pre-reload `inv_freq` tensors which are computed from config and don't change.

Possible causes inside vLLM's layerwise reload:
1. **`_layerwise_process` copy bug**: The `param.data.copy_()` step may not correctly update cudagraph-captured memory for all layers, despite counts showing complete.
2. **`_place_kernel_tensors` on RotaryEmbedding**: Although it should just restore original tensors, it may interact badly with the compiled model state (e.g., invalidating a fused kernel that spans attention + rotary).
3. **Race condition**: The reload may not be fully synchronized before the next inference request replays a cudagraph.
4. **Materialization side effects**: The layerwise reload moves layers to meta device then back. For RotaryEmbedding, `_place_kernel_tensors` restores original tensors, but any intermediate state corruption during the meta→device transition could persist.

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

## Questions for vLLM Team

1. All parametric layers show "Processed" with correct element counts, yet inference produces NaN after the first weight sync. Is there a known issue with `_layerwise_process`'s `param.data.copy_()` not correctly updating cudagraph-captured tensor storage?
2. Could `_place_kernel_tensors` on RotaryEmbedding (which restores pre-reload tensors) corrupt the compiled model state, even though RotaryEmbedding's data hasn't changed?
3. Is there a synchronization gap between `finalize_layerwise_reload()` and the next cudagraph replay?
4. Has `NCCLWeightTransferEngine` been tested for online RLHF-style repeated weight updates (not just one-shot checkpoint loading)?

## Files Changed

- `open_instruct/grpo_fast.py`: `setup_model_update_group()` uses `NCCLWeightTransferEngine.trainer_init()`
- `open_instruct/vllm_utils.py`: `broadcast_weights_to_vllm()` uses `NCCLWeightTransferEngine.trainer_send_weights()`. `LLMRayActor` uses `init_weight_transfer_engine()` and `update_weights()`
- `open_instruct/vllm_utils_workerwrap.py`: Deleted
- Engine creation uses `weight_transfer_config=WeightTransferConfig(backend="nccl")`
