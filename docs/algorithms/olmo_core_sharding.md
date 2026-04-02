# OLMo-core Sharding and Parallelism

The DPO and GRPO trainers in open-instruct use OLMo-core's parallelism primitives for distributed training. This page documents their shared sharding architecture and per-algorithm differences.

## Overview

OLMo-core uses **Hybrid Sharded Data Parallelism (HSDP)** via PyTorch's DTensor-based FSDP2. HSDP combines full sharding within a group of GPUs with replication across groups, balancing memory savings with communication efficiency.

DPO and GRPO share these defaults:

| Setting | Value |
|---------|-------|
| Parallelism type | HSDP (`DataParallelType.hsdp`) |
| Wrapping strategy | `blocks` (each Transformer block is one FSDP unit) |
| Parameter dtype | `bfloat16` |
| Reduction dtype | `float32` |

## How HSDP Works

In HSDP, GPUs are organized into **shard groups** and **replica groups**:

- **Shard group**: A set of GPUs that collectively hold one full copy of the model. Parameters and gradients are sharded (split) across these GPUs, reducing per-GPU memory.
- **Replica group**: Independent shard groups that each hold a full copy. Gradients are all-reduced across replicas after each step.

For example, with 16 GPUs, `shard_degree=8`, and `num_replicas=2`:

- GPUs 0-7 form shard group 1, GPUs 8-15 form shard group 2.
- Each group holds a complete model, sharded across 8 GPUs.
- After the backward pass, gradients are reduced across the two groups.

When `shard_degree` and `num_replicas` are both `None` (the default), OLMo-core auto-detects using node boundaries: `num_replicas` is set to the number of nodes and `shard_degree` is set to the number of GPUs per node. This means all-gather/reduce-scatter traffic stays within a single node (fast NVLink), while only the lighter all-reduce crosses the network between nodes.

If only one of the two is specified, OLMo-core infers the other by dividing `dp_world_size` by the given value. Both values must evenly divide the data-parallel world size.

See [`DataParallelConfig.get_replicate_and_shard_degree()`](https://github.com/allenai/OLMo-core/blob/ffe2eec45a2010936575cc06dc7f6f2a81548872/src/olmo_core/distributed/parallel/data_parallel.py#L41) for the implementation.

### Wrapping Strategy

Both trainers use the `blocks` wrapping strategy, which wraps each Transformer block as an individual FSDP unit. This provides a good balance between:

- **Communication overhead**: Fewer FSDP units means fewer all-gather/reduce-scatter operations.
- **Memory efficiency**: Each block's parameters can be gathered and freed independently.

## Per-Algorithm Configuration

### DPO

DPO (`open_instruct/dpo.py`) provides the most explicit control over sharding:

```python
dp_config = TransformerDataParallelConfig(
    name=DataParallelType.hsdp,
    num_replicas=args.fsdp_num_replicas,   # None = auto
    shard_degree=args.fsdp_shard_degree,   # None = auto
    param_dtype=DType.bfloat16,
    reduce_dtype=DType.float32,
    wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
)
```

**DPO-specific flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--fsdp_shard_degree` | Number of GPUs per shard group | `None` (auto) |
| `--fsdp_num_replicas` | Number of replica groups | `None` (auto) |
| `--tensor_parallel_degree` | Tensor parallelism degree | `1` (disabled) |
| `--context_parallel_degree` | Context (sequence) parallelism degree | `1` (disabled) |

DPO also supports **tensor parallelism** (splitting individual layers across GPUs), which can be combined with HSDP for very large models. Context parallelism (`--context_parallel_degree`) is not yet supported in DPO.

### GRPO

GRPO (`open_instruct/grpo_olmo_core_actor.py`) uses the same HSDP configuration but relies on auto-detection for shard degree and replicas:

```python
dp_config = None
if not single_gpu_mode and world_size > 1:
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        param_dtype=olmo_core_dtype,       # bfloat16 or float32
        reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
    )
```

Key differences from DPO:

- **No explicit shard degree / replica flags.** GRPO always auto-detects.
- **Single-GPU mode.** When `--single_gpu_mode` is set, `dp_config` is `None` and no sharding is applied. The model is instead cast to the target dtype directly.
- **Ray actors.** GRPO uses Ray to coordinate distributed training, so `torch.distributed` is initialized within each Ray actor rather than at the script level.

## Comparison

DPO and GRPO share HSDP with `blocks` wrapping and `float32` reductions. The table below shows where they differ:

| Aspect | DPO | GRPO |
|--------|-----|------|
| Explicit shard degree / replicas | Yes | No (auto) |
| Tensor parallelism | Yes | No |
| Context parallelism | Not yet supported | No |
| Activation checkpointing | Budget-mode | Gradient checkpointing flag |
| Training coordinator | `torch.distributed` | Ray actors |

## Choosing Parallelism Settings

For most use cases, leaving `fsdp_shard_degree` and `fsdp_num_replicas` as `None` (auto-detect) works well. Manually tune these when:

- **You need to control communication patterns.** For example, setting `shard_degree` to the number of GPUs per node ensures all-gather/reduce-scatter stays within a single node (fast NVLink), while cross-node communication is limited to all-reduce across replicas.
- **You're hitting OOM.** Increasing `shard_degree` (more GPUs per shard group) reduces per-GPU memory at the cost of more communication.
- **You're scaling to many nodes.** For large clusters, explicitly setting `num_replicas` can prevent OLMo-core from choosing a suboptimal layout.

For very large models (32B+), consider enabling tensor parallelism (`--tensor_parallel_degree`) in DPO to keep per-GPU memory manageable.

## FSDP-First Loading Pattern

DPO and GRPO follow the same model initialization sequence:

1. Build the OLMo-core `Transformer` model on CPU.
2. Create the `TrainModule`, which calls `parallelize_model()` internally. This applies FSDP sharding and reinitializes weights from scratch.
3. Reload the HuggingFace checkpoint into the now-sharded model via `load_hf_model()`.

This "FSDP-first" pattern ensures that FSDP metadata is set up before any real weights are loaded, avoiding the need to hold a full unsharded copy in memory. See `DPOTrainModule` in `open_instruct/olmo_core_train_modules.py` and `PolicyTrainerOLMoCoreProcess.setup_model()` in `open_instruct/grpo_olmo_core_actor.py` for the DPO and GRPO implementations respectively.
