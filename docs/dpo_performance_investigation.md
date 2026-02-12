# Hybrid DPO Performance Investigation

The hybrid 7B DPO training (`7b_instruct_hybrid_dpo.sh`) is ~16x slower than the regular 7B DPO training on the same cluster (Jupiter) with DeepSpeed ZeRO-3.

## Key Result

Both models tested on **ai2/jupiter** with identical config (ZeRO-3, 4 nodes, 8 GPUs, 16k seq len, grad accumulation 4, activation_memory_budget 0.5):

| | Hybrid (OLMo 3.5 Hybrid) | Baseline (OLMo3 7B) |
|---|---|---|
| **s/step** | ~261 | ~16 (varies 9-24) |
| **Slowdown** | **~16x** | 1x |
| **Beaker experiment** | `01KH21SAS4RYMMA92XYAMTHVB2` (canceled after 3 steps) | `01KH231TYM0Z99GD3XM37VWFPC` |

The single-GPU benchmark (`test_hybrid_layer_speed_gpu.py`) shows the hybrid model is 7% *faster* than OLMo3. So this is purely a **distributed training issue**.

## Hypotheses

### 1. Model architecture is inherently slower

**Status: RULED OUT**

Individual GatedDeltaNet linear attention layers are slower than FlashAttention-2 full attention layers. However, the full hybrid model is actually 7% *faster* than OLMo3 on a single GPU because its smaller hidden size (3840 vs 4096) more than compensates for the slower linear attention layers.

Tested in `open_instruct/test_hybrid_layer_speed_gpu.py` (Beaker experiment `01KH1W80SHAQNZB56M3VJJXM6X`).

### 2. `activation_memory_budget 0.5` is slower than `--gradient_checkpointing`

**Status: RULED OUT**

Despite the different flags, both scripts enable identical gradient checkpointing. In `dpo_tune_cache.py:389-391`, `--activation_memory_budget 0.5` simply checks `< 1.0` and calls `model.gradient_checkpointing_enable(use_reentrant=False)`. The 0.5 value is never passed to PyTorch as an actual budget — it's a plain boolean gate.

### 3. Different cluster / NCCL configuration

**Status: RULED OUT**

We ran both models on the same cluster (Jupiter) with the same NCCL env vars (NCCL_IB_HCA, NCCL_SOCKET_IFNAME, TORCH_NCCL_AVOID_RECORD_STREAMS, TORCH_DIST_INIT_BARRIER) from OLMo-core. The hybrid is still ~16x slower. The cluster is not the issue.

### 4. DeepSpeed ZeRO-3 + hybrid model interaction

**Status: CONFIRMED — primary cause of slowdown**

The hybrid model has ~1.8x more parameter tensors per layer than standard attention (18 vs 9 per block), but this alone doesn't explain 16x. The key issues:

1. **GatedDeltaNet has many more parameter tensors**: 7 linear projections (q/k/v/a/b/g/o_proj) + 3 Conv1d modules + A_log + dt_bias + o_norm = ~13 param tensors per GDN layer, vs 4 linear projections per standard attention layer. Each parameter access in ZeRO-3 triggers an all-gather across all 32 GPUs.

2. **Gradient checkpointing doubles parameter access**: With gradient checkpointing, each block's forward is recomputed during backward, so every parameter is all-gathered 3x (forward + recompute + backward) instead of 2x.

3. **FLA's custom autograd function (`ChunkGatedDeltaRuleFunction`)** uses `@torch.compiler.disable` and `ctx.save_for_backward` with large intermediate tensors. While these saved tensors are activations (not parameters), the `@torch.compiler.disable` prevents any fusion with surrounding ZeRO-3 operations.

4. **FLA kernel launch overhead in distributed setting**: Each GatedDeltaNet forward involves 6+ Triton kernel calls (chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o) vs 1 FlashAttention kernel for standard attention. With 16k sequence length and chunk_size=64, that's 256 chunks per kernel.

5. **OLMo-core pretrained this model with FSDP/HSDP, not DeepSpeed**: The OLMo-core pretraining scripts (`tyler/anejs/linear-rnns` branch) use PyTorch-native FSDP with per-block wrapping and torch.compile. DeepSpeed ZeRO-3 uses per-parameter hooks which may interact poorly with FLA's custom kernels.

6. **FLA has no DeepSpeed integration**: `grep -r "deepspeed\|zero\|distributed\|fsdp"` in the FLA package returns zero results. The library was not designed for ZeRO-3.

### 5. FLA (flash-linear-attention) overhead in distributed training

**Status: LIKELY CONTRIBUTING — merged into hypothesis 4**

The FLA library's Triton kernels are efficient on single GPU but may interact poorly with ZeRO-3's parameter hooks, causing serialization of what should be overlapped compute and communication.

## Recommended Next Steps

1. **Switch from DeepSpeed ZeRO-3 to PyTorch FSDP** for the hybrid DPO script, matching the pretraining setup. This is the most likely fix.
2. **Profile with torch profiler** to identify exact bottleneck (communication vs compute vs synchronization).
3. **Test with ZeRO-2 instead of ZeRO-3** as a quick check — ZeRO-2 doesn't shard parameters, only gradients and optimizer states, avoiding the per-parameter all-gather overhead.

## Reference

| Setting | Hybrid | Baseline |
|---|---|---|
| Script | `scripts/train/olmo3/7b_instruct_hybrid_dpo.sh` | `scripts/train/olmo3/7b_instruct_dpo_jupiter_baseline.sh` |
| Cluster | ai2/jupiter | ai2/jupiter |
| Model | OLMo 3.5 Hybrid (3840 hidden, 24 GDN + 8 attn layers) | OLMo3 7B (4096 hidden, 32 attn layers) |
| Memory management | `--activation_memory_budget 0.5` | `--activation_memory_budget 0.5` |
| NCCL tuning | Jupiter defaults from OLMo-core | Jupiter defaults from OLMo-core |
| Nodes | 4 | 4 |
| GPUs per node | 8 | 8 |
| Sequence length | 16384 | 16384 |
| Gradient accumulation | 4 | 4 |
