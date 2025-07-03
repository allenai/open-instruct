# GRPO Actor Creation in open_instruct/grpo_fast.py

This document explains how Ray actors are created and managed in the GRPO (Group Relative Policy Optimization) implementation.

## Overview

The GRPO implementation uses Ray actors to distribute training across multiple GPUs and nodes. The actor system consists of:
- A master actor (rank 0) that coordinates distributed training
- Worker actors (ranks 1 to world_size-1) that participate in training
- Placement groups to ensure proper resource allocation across nodes

## Actor Creation Process

### 1. Ray Initialization (line 1508)
```python
ray.init(dashboard_host="0.0.0.0")  # enable debugging from a different machine
```
Ray is initialized with the dashboard exposed for remote debugging capabilities.

### 2. Placement Group Setup (lines 1510-1512)
```python
bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
pg = placement_group(bundles, strategy="STRICT_SPREAD")
ray.get(pg.ready())
```
- Creates resource bundles based on `num_learners_per_node` configuration
- Each bundle specifies GPU and CPU requirements
- Uses "STRICT_SPREAD" strategy to distribute actors across different nodes
- Waits for placement group to be ready before proceeding

### 3. Actor Class Definition (lines 484-485)
```python
@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
```
- Decorated with `@ray.remote(num_gpus=1)` to make it a Ray actor
- Inherits from `RayProcess` base class
- Contains methods for model training, weight updates, and synchronization

### 4. ModelGroup Creation (lines 991-1044)

The `ModelGroup` class manages the collection of actor instances:

```python
class ModelGroup:
    def __init__(self, pg, ray_process_cls, num_gpus_per_node, single_gpu_mode):
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
```

Key resource allocations:
- **Single GPU mode**: 0.48 GPUs per actor (allows multiple actors on one GPU)
- **Multi-GPU mode**: 1 GPU per actor
- **CPUs**: 4 CPUs per actor regardless of mode

### 5. Master Actor Creation (lines 1006-1015)
```python
master_policy = ray_process_cls.options(
    num_cpus=self.num_cpus_per_actor,
    num_gpus=self.num_gpus_per_actor,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=self.pg, placement_group_bundle_index=0
    ),
).remote(world_size, 0, 0, None, None)

master_addr, master_port = ray.get(master_policy.get_master_addr_port.remote())
```
- Master actor is created first with rank 0
- Placed in bundle index 0 of the placement group
- Establishes the communication endpoint for distributed training

### 6. Worker Actor Creation (lines 1033-1044)
```python
for rank in range(1, world_size):
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=self.pg,
        placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
    )
    worker_policy = ray_process_cls.options(
        num_cpus=self.num_cpus_per_actor,
        num_gpus=self.num_gpus_per_actor,
        scheduling_strategy=scheduling_strategy,
    ).remote(world_size, rank, 0, master_addr, master_port)
```
- Creates worker actors for ranks 1 through world_size-1
- Uses `get_bundle_index()` to determine placement group bundle
- Passes master address/port for distributed training coordination

### 7. Model Initialization (lines 1521-1524)
```python
inits.extend(
    model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
    for model in policy_group.models
)
```
- After actor creation, models are initialized on each actor
- Loads pretrained weights and sets up training infrastructure

## Key Design Features

1. **Resource Isolation**: Placement groups ensure actors are properly distributed across nodes
2. **Flexible GPU Allocation**: Single GPU mode allows multiple actors per GPU for debugging
3. **Distributed Training**: Master-worker pattern enables distributed training coordination
4. **Asynchronous Operations**: Ray's remote calls allow non-blocking model updates

## Bundle Index Calculation

The `get_bundle_index()` function (lines 1017-1023) maps ranks to placement group bundles:
```python
def get_bundle_index(rank, num_gpus_per_node):
    bundle_idx = 0
    while rank >= num_gpus_per_node[bundle_idx]:
        rank -= num_gpus_per_node[bundle_idx]
        bundle_idx += 1
    return bundle_idx
```

This ensures actors are distributed according to the GPU topology specified in `num_learners_per_node`.

## Usage Example

With `num_learners_per_node=[7, 8, 4]`:
- Ranks 0-6 → Bundle 0 (7 GPUs)
- Ranks 7-14 → Bundle 1 (8 GPUs)  
- Ranks 15-18 → Bundle 2 (4 GPUs)

This allows flexible multi-node training configurations based on available hardware.