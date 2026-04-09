# Failed vLLM Native Weight Sync Investigation

**Date**: 2026-04-08
**Branch**: `finbarr/vllm-weight-sync`
**vLLM version**: 0.19.0
**Model**: Qwen/Qwen2.5-7B
**Config**: 4 vLLM engines x TP=2, DeepSpeed stage 3, 2 nodes x 8 GPUs

## Summary

We migrated from a custom `WorkerWrap`-based weight sync (using `torch.distributed.broadcast` + `model_runner.model.load_weights`) to vLLM's native `NCCLWeightTransferEngine` API. After the first training step's weight sync, all vLLM engines produce NaN outputs and crash.

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

**Key finding**: With DEBUG logging, all parametric layers successfully receive and apply weights:

```
VocabParallelEmbedding: 272498688 / 272498688 → Processed
QKVParallelLinear: 8259840 / 8259840 → Processed      (q_proj + k_proj + v_proj + biases, fused)
RowParallelLinear: 6422528 / 6422528 → Processed       (o_proj)
MergedColumnParallelLinear: 67895296 / 67895296 → Processed  (gate_proj + up_proj, fused)
RowParallelLinear: 33947648 / 33947648 → Processed     (down_proj)
RMSNorm: 3584 / 3584 → Processed
ParallelLMHead: 272498688 / 272498688 → Processed
```

The ONLY module that fails is **RotaryEmbedding**. The patch suppressed warnings for truly zero-parameter modules (SiluAndMul, ApplyRotaryEmb, etc.), but RotaryEmbedding has `load_numel_total > 0` because vLLM counts its `inv_freq` buffer as a weight.

## Root Cause Analysis

### Why RotaryEmbedding fails

RotaryEmbedding has an `inv_freq` buffer (not a trainable parameter). We only send `model.named_parameters()` over NCCL, so `inv_freq` is never sent. vLLM's layerwise loader sees `load_numel_total > 0` (from `inv_freq`) but `load_numel == 0` (nothing received), triggering:

```python
# layerwise.py:228-230
logger.warning("%s: Failed to load weights", layer.__class__.__name__)
_place_kernel_tensors(layer, info)
```

`_place_kernel_tensors` replaces the RotaryEmbedding's `inv_freq` with compiled kernel tensors, corrupting position embeddings and causing NaN output.

### Why the old approach didn't have this problem

The old `WorkerWrap` approach called `model_runner.model.load_weights(weights=[(name, weight)])` for each parameter individually. This never triggered the layerwise reload path — it directly set weights via the model's `load_weights` method, which only touches the parameters you send and leaves everything else intact.

## How The Old Approach Worked

The old approach used a custom `WorkerWrap` extension class (`vllm_utils_workerwrap.py`):

1. Trainer broadcasts each parameter via `torch.distributed.broadcast(param.data, 0, group=model_update_group)`.
2. Each vLLM worker receives into a fresh tensor via the same broadcast.
3. Worker calls `self.model_runner.model.load_weights(weights=[(name, weight)])` to apply each parameter.

This worked because `load_weights` handles the HF-to-vLLM name mapping and weight fusion (e.g., separate `q_proj`/`k_proj`/`v_proj` → fused `qkv_proj`) internally.

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

The NCCL send completes without error. All parametric layers receive their weights correctly. But `_place_kernel_tensors` is called on RotaryEmbedding (which has an unreceived `inv_freq` buffer), corrupting the model.

## Possible Fixes

1. **Send `inv_freq` as part of the weight update**: Include model buffers (not just parameters) in the NCCL transfer. This would satisfy RotaryEmbedding's expected weight count.
2. **Suppress `_place_kernel_tensors` for RotaryEmbedding**: Extend the Dockerfile patch to also gate on module type or skip non-parameter buffers.
3. **Report to vLLM team**: The `NCCLWeightTransferEngine` reload path shouldn't call `_place_kernel_tensors` for modules whose only "weights" are non-trainable buffers like `inv_freq`. This is a vLLM bug separate from #38574.

## Questions for vLLM Team

1. Should `_place_kernel_tensors` be gated on whether the unreceived weights are trainable parameters vs. non-trainable buffers?
2. Is RotaryEmbedding's `inv_freq` expected to be sent during weight transfer, or should the reload path preserve it from the existing model state?
3. Is there a recommended example of using `NCCLWeightTransferEngine` for online weight updates (not just checkpoint loading)?

## Files Changed

- `open_instruct/grpo_fast.py`: `setup_model_update_group()` now uses `NCCLWeightTransferEngine.trainer_init()` instead of custom `init_process_group()`.
- `open_instruct/vllm_utils.py`: `broadcast_weights_to_vllm()` now uses `NCCLWeightTransferEngine.trainer_send_weights()` instead of `torch.distributed.broadcast()` + `_send_to_vllm()`. Removed custom `init_process_group()`. `LLMRayActor` now uses `init_weight_transfer_engine()` and `update_weights()` instead of `init_process_group()` and `update_weights_batch()`.
- `open_instruct/vllm_utils_workerwrap.py`: Deleted (was the custom worker extension).
- Engine creation now uses `weight_transfer_config=WeightTransferConfig(backend="nccl")` instead of `worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap"`.
- `Dockerfile`: Added hotfix patch for layerwise.py (experiment 2 only).
