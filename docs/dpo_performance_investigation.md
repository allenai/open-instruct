# Hybrid DPO Performance Investigation

The hybrid 7B DPO training (`7b_instruct_hybrid_dpo.sh`) is ~3.5x slower than the regular 7B DPO training (`7b_instruct_dpo.sh`). This document tracks hypotheses and results.

## Hypotheses

### 1. Model architecture is inherently slower

**Status: RULED OUT**

Individual GatedDeltaNet linear attention layers are slower than FlashAttention-2 full attention layers. However, the full hybrid model is actually 7% *faster* than OLMo3 on a single GPU because its smaller hidden size (3840 vs 4096) more than compensates for the slower linear attention layers.

Tested in `open_instruct/test_hybrid_layer_speed_gpu.py` (Beaker experiment `01KH1W80SHAQNZB56M3VJJXM6X`).

### 2. `activation_memory_budget 0.5` is slower than `--gradient_checkpointing`

**Status: RULED OUT**

Despite the different flags, both scripts enable identical gradient checkpointing. In `dpo_tune_cache.py:389-391`, `--activation_memory_budget 0.5` simply checks `< 1.0` and calls `model.gradient_checkpointing_enable(use_reentrant=False)`. The 0.5 value is never passed to PyTorch as an actual budget — it's a plain boolean gate.

The regular script's `--gradient_checkpointing` flag is not even a recognized argument in `dpo_utils.ExperimentConfig`. It's likely silently consumed by `HfArgumentParser` or causes an error. Either way, the checkpointing behavior is equivalent.

### 3. Different cluster / NCCL configuration

**Status: INCONCLUSIVE (cannot test)**

Both clusters use NVIDIA H100 80GB HBM3 GPUs with high-bandwidth interconnect. However:

- **ai2/jupiter** (hybrid run): On-prem AI2 cluster with WEKA shared filesystem. No NCCL transport tuning. Currently active (128 nodes).
- **ai2/augusta** (regular run): GCP-based cluster with tcpxo (Google's optimized NCCL transport using LL128 protocol). No shared filesystem (uses GCS upload/download). **Currently offline (0 nodes, decommissioned).**

The tcpxo NCCL transport on Augusta could improve allgather/reduce-scatter performance in ZeRO-3, but Augusta is offline so we cannot re-run the baseline there. To isolate this variable, we would need to run the regular (non-hybrid) DPO on Jupiter and compare throughput to the original Augusta run.

### 4. DeepSpeed ZeRO-3 + hybrid model interaction

**Status: LIKELY CAUSE — hybrid model was designed for FSDP, not DeepSpeed**

The OLMo-core pretraining scripts for the hybrid model (on the `tyler/anejs/linear-rnns` branch) use **PyTorch-native FSDP/HSDP**, not DeepSpeed ZeRO-3:

| Setting | OLMo-core pretraining | Our DPO script |
|---|---|---|
| Sharding strategy | HSDP (PyTorch native) | DeepSpeed ZeRO-3 |
| Wrapping granularity | Per transformer block | Automatic (DeepSpeed) |
| torch.compile | Enabled | Not used |
| Long-context (65K) phase | FSDP + context parallelism + activation checkpointing (budget 0.1) | N/A |

The hybrid model was designed and optimized for FSDP/HSDP with per-block wrapping. DeepSpeed ZeRO-3 uses a fundamentally different parameter-gathering approach (hooks-based, per-parameter) that may interact poorly with:
- GatedDeltaNet's custom parameter shapes (head_dim=96, value_head_dim=192, conv kernel dim=4)
- FLA's custom CUDA kernels during backward pass
- The lack of per-block wrapping granularity control in DeepSpeed

Our DPO ZeRO-3 config (`stage3_no_offloading_accelerate.conf`) uses generic settings with no hybrid-specific tuning:
- `sub_group_size: 1e9`, `stage3_max_live_parameters: 1e9`, `overlap_comm: true`
- No selective gathering of linear attention parameters
- No custom bucket sizes tuned to GatedDeltaNet parameter structure

**Recommended next step:** Try switching the hybrid DPO script from DeepSpeed ZeRO-3 to PyTorch-native FSDP to match the pretraining setup.

### 5. FLA (flash-linear-attention) overhead in distributed training

**Status: NOT YET TESTED**

The FLA library may have overhead in multi-GPU/multi-node settings that doesn't appear in single-GPU benchmarks (e.g., synchronization barriers, incompatible CUDA graphs, etc.).

## Reference

| Setting | Hybrid (slow) | Regular (fast) |
|---|---|---|
| Script | `scripts/train/olmo3/7b_instruct_hybrid_dpo.sh` | `scripts/train/olmo3/7b_instruct_dpo.sh` |
| Cluster | `ai2/jupiter` | `ai2/augusta` |
| Memory management | `--activation_memory_budget 0.5` | `--gradient_checkpointing` |
| NCCL tuning | None | tcpxo (LL128, tuner config) |
| Model hidden size | 3840 | 4096 |
| Nodes | 4 | 4 |
| GPUs per node | 8 | 8 |
| Regular DPO Beaker experiment | | `01KA62AJW9P8AWA3YKWE4Y6XZD` |
