# DPO MFU Optimization Plan

## Context

Baseline run:

- Beaker: `01KT58SPP2MVYN4EAKYHZN3DN6`
- W&B: https://wandb.ai/ai2-llm/open_instruct_internal/runs/gzhp5mp7
- Script: `scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh`
- Model: `allenai/Olmo-Hybrid-Instruct-SFT-7B`
- Objective: improve DPO training MFU from the observed ~8.5%.

Baseline configuration:

- 4 nodes, 8 GPUs per node, 32 GPUs total
- `--max_seq_length 16384`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 4`
- `--fsdp_shard_degree 32`
- `--fsdp_num_replicas 1`
- `--packing`
- `--activation_checkpointing_mode selected_modules`
- `--compile_model true`

Runtime observations from Beaker logs:

- Data loading is not the bottleneck: about 0.1-0.2% of wall time.
- GPU active memory is about 52 GiB, leaving meaningful headroom on 80 GiB GPUs.
- Steady-state wall time is about 3.6 seconds per optimizer step.
- Real token throughput varies heavily across steps, indicating uneven packed-token occupancy.

The main bet is that the run is doing too much fixed-shape compute and cross-node communication for too few useful tokens.

## Hypotheses

1. Packed-token occupancy is too low.

   The DPO data loader uses `per_device_train_batch_size * gradient_accumulation_steps` as the per-rank candidate limit for packing. With the baseline values, each rank only considers 4 examples when building a 16k chosen / 16k rejected packed row. The model still runs over the padded packed shape, so underfilled packs waste compute.

2. Full 32-way sharding is too communication-heavy.

   The baseline uses one FSDP shard group over all 32 GPUs. For a 7B model with 52 GiB active memory, smaller shard groups with HSDP replication may reduce cross-node all-gather/reduce-scatter overhead while still fitting in memory.

3. Activation checkpointing may be more aggressive than needed.

   `selected_modules` checkpointing saves memory but adds recompute. Since the baseline has memory headroom, a lighter checkpointing mode, or no checkpointing, may improve step time.

## Experiment Matrix

Run short A/B experiments first. Each run only needs enough steady-state steps after compile and cache warmup to compare throughput, e.g. 100-200 training steps after training starts.

| Run | Goal | Key changes | Expected outcome | Main risk |
| --- | --- | --- | --- | --- |
| Baseline repeat | Confirm current behavior on same image/code | Original config | MFU around 8.5%; ~0.28 device-BPS | Cluster noise |
| Pack candidates 16 | Improve useful-token occupancy | `GRADIENT_ACCUMULATION_STEPS=16` | More real tokens per step, higher MFU | Larger effective batch changes optimization |
| HSDP 8x4 | Reduce cross-node FSDP communication | `FSDP_SHARD_DEGREE=8`, `FSDP_NUM_REPLICAS=4` | Faster step time at similar token count | OOM or worse memory pressure |
| HSDP 4x8 | More aggressive communication reduction | `FSDP_SHARD_DEGREE=4`, `FSDP_NUM_REPLICAS=8` | Faster than 8x4 if memory allows | Higher OOM risk |
| No selected-module AC | Reduce recompute | `ACTIVATION_CHECKPOINTING_MODE=budget` with default budget | Faster step time if memory fits | OOM or compile incompatibility |
| Combined best | Validate interaction effects | Best packing + best HSDP + best AC mode | Highest MFU candidate | Interactions may differ from isolated runs |

## Launch Commands

Assuming the script keeps env-var overrides for the tuning knobs:

```bash
# 1. Baseline repeat
EXP_TAG=-baseline \
./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh

# 2. More examples considered per packed row
GRADIENT_ACCUMULATION_STEPS=16 \
EXP_TAG=-pack16 \
./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh

# 3. Per-node shard groups, 4 replicas
FSDP_SHARD_DEGREE=8 \
FSDP_NUM_REPLICAS=4 \
EXP_TAG=-hsdp8x4 \
./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh

# 4. Smaller shard groups, 8 replicas
FSDP_SHARD_DEGREE=4 \
FSDP_NUM_REPLICAS=8 \
EXP_TAG=-hsdp4x8 \
./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh

# 5. Lighter activation checkpointing
ACTIVATION_CHECKPOINTING_MODE=budget \
EXP_TAG=-budget-ac \
./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core.sh
```

Per repo workflow, `build_image_and_launch.sh` requires committed changes. Commit the script/doc changes before launching experiments.

## Metrics To Compare

Track these from W&B and Beaker logs:

- `perf/mfu_step`
- `perf/mfu_avg`
- `perf/tokens_per_second_step`
- `perf/tokens_per_second_per_gpu`
- `perf/seconds_per_step`
- `perf/data_loading_pct`
- `throughput/device/BPS`
- `throughput/device/BPS (actual avg)`
- `throughput/total tokens`
- `gpu_memory/GPU active mem (GiB)`
- `gpu_memory/GPU reserved mem (GiB)`
- Loss curve and reward metrics, especially if changing effective batch size.

For packing experiments, compute useful-token occupancy:

```text
occupancy = real_tokens_per_step / (num_gpus * 2 * max_seq_length)
```

For the baseline, the denominator is:

```text
32 * 2 * 16384 = 1,048,576 tokens per step
```

The baseline logs showed roughly 200k real tokens per step in a recent window, or about 20% occupancy.

## Decision Rules

1. If `GRADIENT_ACCUMULATION_STEPS=16` materially increases token occupancy and MFU without hurting loss behavior, keep it or sweep nearby values such as 8, 16, and 32.

2. If `FSDP_SHARD_DEGREE=8, FSDP_NUM_REPLICAS=4` improves step time without OOM, prefer it over 32-way sharding. Try 4x8 only if 8x4 is stable and memory still has headroom.

3. If `ACTIVATION_CHECKPOINTING_MODE=budget` fits and improves step time, keep it. If it OOMs, try a budgeted run with an explicit memory budget before returning to `selected_modules`.

4. Once the best individual knobs are identified, run a combined experiment and compare against the baseline repeat, not only against the original run.

5. Do not judge by MFU alone. A configuration that raises MFU by changing effective batch size still needs a sanity check on loss, reward margin, and downstream eval plan.

## Possible Code Follow-Up

The cleaner long-term fix is to decouple packing candidate count from optimizer batch semantics. Today, `gradient_accumulation_steps` controls how many examples the packing loader considers per rank, but with padding-free DPO the packed batch is still one row and `split_batch_dpo()` usually does not create multiple backward microbatches.

Add a separate argument such as `--packing_max_examples_per_rank` or `--packing_candidate_multiplier`, then use that in `HFDataLoader` for packing while preserving the intended optimizer batch size. This would let us improve token occupancy without changing DPO effective batch size.
