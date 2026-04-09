# Failed vLLM Native Weight Sync Investigation

**Date**: 2026-04-08
**Branch**: `finbarr/vllm-weight-sync`
**Experiment**: [01KNQB2A4APPPHK19CWNVGHK7F](https://beaker.org/ex/01KNQB2A4APPPHK19CWNVGHK7F) (failed)
**W&B run**: [xqt73bw0](https://wandb.ai/ai2-llm/open_instruct_internal/runs/xqt73bw0)
**vLLM version**: 0.19.0
**Model**: Qwen/Qwen2.5-7B
**Config**: 4 vLLM engines x TP=2, DeepSpeed stage 3, 2 nodes x 8 GPUs

## Summary

We migrated from a custom `WorkerWrap`-based weight sync (using `torch.distributed.broadcast` + `model_runner.model.load_weights`) to vLLM's native `NCCLWeightTransferEngine` API. After the first training step's weight sync, all vLLM engines fail to apply the received weights, producing NaN outputs and crashing.

## Timeline

| Time | Event | Status |
|------|-------|--------|
| 20:12:44 | Step 0: initial inference with original weights | OK |
| 20:15:52 | Training step 1 starts | OK |
| 20:16:15 | Training step 1 completes (grad_norm: 7.6e9) | OK |
| 20:16:16 | Weight sync completes (0.71s, no NCCL errors) | OK |
| 20:16:16 | `layerwise.py:230` "Failed to load weights" for all modules | FAIL |
| 20:16:22 | vLLM inference returns NaN | FAIL |
| 20:16:43 | Experiment crashes | FAIL |

## Root Cause

The NCCL transport succeeds (no errors, completes in 0.71s), but vLLM's internal layerwise weight loader fails to apply the received weights to the model. Every module type reports failure:

```
WARNING [layerwise.py:230] Qwen2ForCausalLM: Failed to load weights
WARNING [layerwise.py:230] Qwen2Model: Failed to load weights
WARNING [layerwise.py:230] ModuleList: Failed to load weights
WARNING [layerwise.py:230] Qwen2DecoderLayer: Failed to load weights
WARNING [layerwise.py:230] Qwen2Attention: Failed to load weights
WARNING [layerwise.py:230] RotaryEmbedding: Failed to load weights
WARNING [layerwise.py:230] ApplyRotaryEmb: Failed to load weights
WARNING [layerwise.py:230] SiluAndMul: Failed to load weights
WARNING [layerwise.py:230] LogitsProcessor: Failed to load weights
```

Both TP0 and TP1 workers on all 4 engines fail. The model then runs inference with stale/uninitialized weights, producing NaN.

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

The NCCL send completes without error, but `layerwise.py:230` reports "Failed to load weights" for every module on every TP worker.

## Hypothesis

The `NCCLWeightTransferEngine` receives the weight data over NCCL correctly, but the layerwise application step inside vLLM fails. Possible causes:

1. **Name mismatch**: HF parameter names (e.g., `model.layers.0.self_attn.q_proj.weight`) may not match what vLLM's layerwise loader expects. The old `load_weights` method handled this mapping; the new engine may not.
2. **Weight fusion**: vLLM fuses weights internally (e.g., `q_proj`+`k_proj`+`v_proj` → `qkv_proj`, `gate_proj`+`up_proj` → `gate_up_proj`). The old approach sent individual HF weights and `load_weights` handled fusion. The new NCCL engine may expect pre-fused weights.
3. **TP sharding**: With TP=2, each worker needs a different shard. The old approach broadcast full weights and `load_weights` handled sharding. The new engine's handling of TP sharding during weight application may be broken.
4. **`packed=False` code path**: We use `packed=False`. There may be a bug in vLLM's unpacked weight application for the NCCL backend.

## Questions for vLLM Team

1. Does `NCCLWeightTransferEngine` handle the HF → vLLM weight name mapping and fusion internally, or does the caller need to send pre-mapped/pre-fused weights?
2. Is there a known issue with `packed=False` and TP>1?
3. What does the "Failed to load weights" warning from `layerwise.py:230` specifically indicate? Is it always a failure, or can it be benign for non-parametric modules?
4. Is there a recommended example of using `NCCLWeightTransferEngine` for online weight updates (not just checkpoint loading)?

## Files Changed

- `open_instruct/grpo_fast.py`: `setup_model_update_group()` now uses `NCCLWeightTransferEngine.trainer_init()` instead of custom `init_process_group()`.
- `open_instruct/vllm_utils.py`: `broadcast_weights_to_vllm()` now uses `NCCLWeightTransferEngine.trainer_send_weights()` instead of `torch.distributed.broadcast()` + `_send_to_vllm()`. Removed custom `init_process_group()`. `LLMRayActor` now uses `init_weight_transfer_engine()` and `update_weights()` instead of `init_process_group()` and `update_weights_batch()`.
- `open_instruct/vllm_utils_workerwrap.py`: Deleted (was the custom worker extension).
- Engine creation now uses `weight_transfer_config=WeightTransferConfig(backend="nccl")` instead of `worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap"`.
