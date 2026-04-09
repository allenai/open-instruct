# Failed vLLM Native Weight Sync Investigation

**Date**: 2026-04-08 to 2026-04-09
**Branch**: `finbarr/vllm-weight-sync`
**vLLM version**: 0.19.0
**Model**: Qwen/Qwen2.5-7B
**Config**: 4 vLLM engines x TP=2, DeepSpeed stage 3, 2 nodes x 8 GPUs

## Summary

We migrated from a custom `WorkerWrap`-based weight sync (using `torch.distributed.broadcast` + `model_runner.model.load_weights`) to vLLM's native `NCCLWeightTransferEngine` API. After the first training step's weight sync, all vLLM engines produce NaN outputs and crash. Three experiments confirmed this is a vLLM bug in the layerwise reload path.

## Experiments

### Experiment 1: Unpatched vLLM 0.19.0

**Experiment**: [01KNQB2A4APPPHK19CWNVGHK7F](https://beaker.org/ex/01KNQB2A4APPPHK19CWNVGHK7F)
**W&B run**: [xqt73bw0](https://wandb.ai/ai2-llm/open_instruct_internal/runs/xqt73bw0)

| Time | Event | Status |
|------|-------|--------|
| 20:12:44 | Step 0: initial inference with original weights | OK |
| 20:15:52 | Training step 1 starts | OK |
| 20:16:15 | Training step 1 completes (grad_norm: 7.6e9) | OK |
| 20:16:16 | Weight sync completes (0.71s, no NCCL errors) | OK |
| 20:16:16 | `layerwise.py:230` "Failed to load weights" for ALL modules | FAIL |
| 20:16:22 | vLLM inference returns NaN | FAIL |
| 20:16:43 | Experiment crashes | FAIL |

Every module type reports failure, including both parametric modules (QKVParallelLinear, etc.) and non-parametric wrappers (SiluAndMul, RotaryEmbedding, etc.).

### Experiment 2: Patched vLLM 0.19.0 (layerwise.py hotfix + DEBUG logging)

**Experiment**: [01KNSM92NT1WEWVRQF7E24GNGB](https://beaker.org/ex/01KNSM92NT1WEWVRQF7E24GNGB)
**W&B run**: [pv1j3lha](https://wandb.ai/ai2-llm/open_instruct_internal/runs/pv1j3lha)

Applied Dockerfile patch from [vllm-project/vllm#38574](https://github.com/vllm-project/vllm/pull/38574): changed `else:` to `elif info.load_numel_total > 0:` in `layerwise.py:230` to suppress warnings for zero-parameter modules. Enabled `VLLM_LOGGING_LEVEL=DEBUG`.

| Time | Event | Status |
|------|-------|--------|
| 17:27:33 | Step 0: initial inference with original weights | OK |
| 17:30:10 | Training step 1 starts | OK |
| 17:30:33 | Training step 1 completes (grad_norm: 3.4e10) | OK |
| 17:30:34 | Weight sync completes (0.661s, no NCCL errors) | OK |
| 17:30:34 | All parametric layers: "Processed" (weights fully received) | OK |
| 17:30:34 | `RotaryEmbedding: Failed to load weights` (on all engines) | FAIL |
| 17:30:42 | vLLM inference returns NaN | FAIL |
| 17:31:08 | Experiment crashes | FAIL |

All parametric layers successfully receive and apply weights:

```
VocabParallelEmbedding: 272498688 / 272498688 → Processed
QKVParallelLinear: 8259840 / 8259840 → Processed
RowParallelLinear: 6422528 / 6422528 → Processed
MergedColumnParallelLinear: 67895296 / 67895296 → Processed
RowParallelLinear: 33947648 / 33947648 → Processed
RMSNorm: 3584 / 3584 → Processed
ParallelLMHead: 272498688 / 272498688 → Processed
```

The ONLY module that still fails is **RotaryEmbedding**. The patch suppressed warnings for truly zero-parameter modules (SiluAndMul, ApplyRotaryEmb, etc.), but RotaryEmbedding has `load_numel_total > 0` because vLLM counts its `inv_freq` buffer.

### Experiment 3: Sending buffers alongside parameters

**Experiment**: [01KNSNXB533Y6WW4MAHCJDP0XR](https://beaker.org/ex/01KNSNXB533Y6WW4MAHCJDP0XR)
**W&B run**: [pv1j3lha](https://wandb.ai/ai2-llm/open_instruct_internal/runs/pv1j3lha)

Attempted to fix by sending `model.named_buffers()` (including `inv_freq`) alongside `model.named_parameters()` in the NCCL transfer.

| Time | Event | Status |
|------|-------|--------|
| 17:55:30 | Step 0: initial inference with original weights | OK |
| 17:59:02 | Training step 1 completes (grad_norm: 1.41) | OK |
| 17:59:02 | Weight sync completes (0.711s) | OK |
| 17:59:02 | `RotaryEmbedding: Failed to load weights` (still!) | FAIL |
| 17:59:13 | vLLM inference returns NaN | FAIL |
| 17:59:35 | Experiment crashes | FAIL |

**This did not fix the issue.** RotaryEmbedding still reports "Failed to load weights" even when `inv_freq` is included in the transfer. vLLM's `load_weights()` method does not handle `inv_freq` — it's a computed buffer, not a checkpoint weight. The layerwise reload path still sees `load_numel == 0` for RotaryEmbedding and calls `_place_kernel_tensors`, corrupting the model.

Note: grad_norm was 1.41 (healthy), confirming training itself is fine — only the weight sync breaks inference.

## Root Cause

The `NCCLWeightTransferEngine`'s layerwise reload path (`layerwise.py`) calls `_place_kernel_tensors` on RotaryEmbedding after every weight update. RotaryEmbedding has an `inv_freq` buffer that is:
- Counted in `load_numel_total` (vLLM thinks it should have weights)
- Never loaded via `load_weights()` (it's computed from config, not stored in checkpoints)
- Therefore always triggers `load_numel == 0` → `_place_kernel_tensors`

`_place_kernel_tensors` replaces RotaryEmbedding's compiled kernel tensors, corrupting position embeddings and producing NaN.

The old `WorkerWrap` approach called `model_runner.model.load_weights()` directly per parameter, which never triggered the layerwise reload path.

## How The Old Approach Worked

The old approach used a custom `WorkerWrap` extension class (`vllm_utils_workerwrap.py`):

1. Trainer broadcasts each parameter via `torch.distributed.broadcast(param.data, 0, group=model_update_group)`.
2. Each vLLM worker receives into a fresh tensor via the same broadcast.
3. Worker calls `self.model_runner.model.load_weights(weights=[(name, weight)])` to apply each parameter.

This worked because `load_weights` handles the HF-to-vLLM name mapping and weight fusion (e.g., separate `q_proj`/`k_proj`/`v_proj` → fused `qkv_proj`) internally, and never triggers the layerwise reload code path.

## How The New Approach Works (And Fails)

### Engine creation

```python
weight_transfer_config=WeightTransferConfig(backend="nccl")
```

### Initialization (trainer side, rank 0)

```python
self.model_update_group = NCCLWeightTransferEngine.trainer_init(
    {"master_address": addr, "master_port": port, "world_size": world_size}
)
```

### Initialization (vLLM engine side)

```python
request = WeightTransferInitRequest(init_info={
    "master_address": addr, "master_port": port,
    "rank_offset": i * tp_size + 1, "world_size": world_size
})
llm_engine.init_weight_transfer_engine(request)
```

### Weight update

```python
# Step 1: Tell engines which weights to expect
names, dtype_names, shapes = _collect_weight_metadata(model, name_mapper)
refs = [engine.update_weights.remote(names, dtype_names, shapes, packed=False)
        for engine in vllm_engines]

# Step 2: Send actual weight data via NCCL
mapped_params = [(name, param.data) for name, param in model.named_parameters()]
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=iter(mapped_params),
    trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group, packed=False)
)
```

The NCCL send completes without error. All parametric layers receive their weights correctly. But the layerwise reload path calls `_place_kernel_tensors` on RotaryEmbedding, corrupting the model.

## Conclusion

This is a vLLM bug. The `NCCLWeightTransferEngine` reload path should not call `_place_kernel_tensors` on modules whose only "weights" are non-trainable computed buffers like `inv_freq`. PR #38574 fixed the issue for zero-parameter modules but didn't cover this case.

Reported to vLLM team for fix.

## Files Changed

- `open_instruct/grpo_fast.py`: `setup_model_update_group()` now uses `NCCLWeightTransferEngine.trainer_init()` instead of custom `init_process_group()`.
- `open_instruct/vllm_utils.py`: `broadcast_weights_to_vllm()` now uses `NCCLWeightTransferEngine.trainer_send_weights()` instead of `torch.distributed.broadcast()` + `_send_to_vllm()`. Removed custom `init_process_group()`. `LLMRayActor` now uses `init_weight_transfer_engine()` and `update_weights()` instead of `init_process_group()` and `update_weights_batch()`.
- `open_instruct/vllm_utils_workerwrap.py`: Deleted (was the custom worker extension).
- Engine creation now uses `weight_transfer_config=WeightTransferConfig(backend="nccl")` instead of `worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap"`.
