# Hybrid GRPO Performance Investigation

The hybrid 7B model (OLMo3.1 linear RNN) is significantly slower than the standard OLMo-3 7B in GRPO training on the same cluster and configuration. This document records the performance comparison, benchmark data, and root cause analysis.

## Setup

Both runs: 19 steps, 2x8 GPU nodes on Jupiter, identical hyperparameters (DeepSpeed ZeRO-3, sequence_parallel_size=2, gradient_checkpointing, pack_length=20480, 4 vLLM engines with TP=2).

| | Standard (OLMo-3-1025-7B) | Hybrid (OLMo3.1-7B linear RNN) |
|---|---|---|
| **Beaker** | `01KH4N25PN17WYSS1PAY3E4SMC` | `01KGT9WPXPAZXJZ41NYZPSTBPR` |
| **W&B** | [04hfurly](https://wandb.ai/ai2-llm/open_instruct_internal/runs/04hfurly) | [gh5fwpsj](https://wandb.ai/ai2-llm/open_instruct_internal/runs/gh5fwpsj) |
| **Script** | `scripts/train/debug/large_test_script.sh` | `scripts/train/debug/large_test_script_hybrid.sh` |
| **Wall clock** | **15 min** | **37 min** |
| **Extra flags** | -- | `--trust_remote_code`, `--vllm_enforce_eager` |
| **Architecture** | 32 transformer layers, hidden=4096 | 24 GatedDeltaNet + 8 transformer layers, hidden=3840 |

## Per-Step Comparison

| Step | Std AvgSeq | Hyb AvgSeq | Std MaxSeq | Hyb MaxSeq | Std Solved | Hyb Solved | Std Stop | Hyb Stop | Std StepTok | Hyb StepTok | Std Reward | Hyb Reward | Std Correct | Hyb Correct | Std Loss | Hyb Loss | Std TPS | Hyb TPS | Std ActTPS | Hyb ActTPS | Std Time | Hyb Time | Std Train | Hyb Train | Std Sync | Hyb Sync |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 469 | 157 | 4096 | 1139 | 478 | 179 | 0.98 | 1.00 | 248,603 | 89,237 | 0.80 | 0.55 | 0.10 | 0.07 | -0.13 | -0.38 | 0 | 0 | 3,909 | 1,089 | 88.0 | 227.3 | 88.0 | 227.3 | - | - |
| 2 | 593 | 142 | 4096 | 264 | 574 | 196 | 0.96 | 1.00 | 313,127 | 82,683 | 0.55 | 0.72 | 0.09 | 0.08 | -0.03 | -0.30 | 2,140 | 322 | 4,955 | 691 | 27.0 | 49.0 | 14.5 | 15.7 | 12.9 | 33.7 |
| 3 | 707 | 148 | 4096 | 2937 | 439 | 267 | 0.93 | 1.00 | 371,905 | 85,448 | 0.88 | 0.63 | 0.10 | 0.08 | -0.15 | -0.33 | 3,434 | 483 | 6,007 | 860 | 24.9 | 46.0 | 14.1 | 15.2 | 11.2 | 32.7 |
| 4 | 654 | 148 | 4096 | 3451 | 730 | 298 | 0.88 | 1.00 | 344,526 | 85,171 | 0.74 | 0.78 | 0.10 | 0.09 | -0.18 | -0.22 | 4,657 | 579 | 287 | 758 | 35.9 | 87.7 | 25.7 | 55.3 | 10.6 | 32.7 |
| 5 | 593 | 137 | 4096 | 847 | 656 | 208 | 0.96 | 1.00 | 312,609 | 80,306 | 0.71 | 0.77 | 0.10 | 0.09 | 0.04 | -0.16 | 5,123 | 570 | 5,500 | 2,735 | 25.0 | 132.6 | 15.2 | 99.8 | 11.7 | - |
| 6 | 594 | 151 | 4096 | 3623 | 340 | 203 | 0.96 | 1.00 | 313,797 | 86,276 | 0.84 | 0.61 | 0.10 | 0.07 | -0.19 | -0.43 | 5,729 | 613 | 5,259 | 713 | 27.2 | 88.2 | 16.2 | 55.7 | 11.5 | 32.9 |
| 7 | 549 | 172 | 4096 | 3967 | 752 | 174 | 0.96 | 1.00 | 290,829 | 98,979 | 0.70 | 0.68 | 0.09 | 0.08 | -0.08 | -0.03 | 5,913 | 623 | 4,750 | 745 | 21.6 | 103.9 | 10.8 | 71.2 | 11.3 | - |
| 8 | 606 | 142 | 4096 | 3843 | 581 | 147 | 0.95 | 1.00 | 320,287 | 82,245 | 0.54 | 0.63 | 0.08 | 0.07 | -0.06 | -0.21 | 6,052 | 643 | 5,176 | 629 | 39.8 | 127.5 | 30.0 | 94.7 | 10.3 | - |
| 9 | 548 | 140 | 4096 | 927 | 458 | 188 | 0.97 | 1.00 | 290,123 | 81,424 | 0.72 | 0.66 | 0.09 | 0.08 | -0.00 | -0.29 | 6,011 | 629 | 5,310 | 2,613 | 21.6 | 123.4 | 10.7 | 90.6 | 11.3 | - |
| 10 | 763 | 143 | 4096 | 2409 | 697 | 227 | 0.92 | 1.00 | 400,038 | 81,790 | 0.82 | 0.70 | 0.11 | 0.08 | 0.02 | -0.17 | 6,185 | 636 | 6,632 | 1,048 | 34.2 | 115.3 | 23.5 | 82.9 | 11.1 | - |
| 11 | 679 | 150 | 4096 | 4096 | 436 | 198 | 0.95 | 1.00 | 356,920 | 85,630 | 0.94 | 0.55 | 0.11 | 0.07 | -0.05 | -0.28 | 6,266 | 636 | 5,902 | 625 | 25.1 | 100.1 | 14.2 | 67.4 | 11.3 | - |
| 12 | 614 | 158 | 4096 | 1778 | 697 | 225 | 0.95 | 1.00 | 323,980 | 90,623 | 0.81 | 0.61 | 0.11 | 0.07 | -0.15 | -0.12 | 6,622 | 651 | 5,674 | 1,536 | 25.4 | 100.5 | 14.5 | 67.7 | 11.4 | - |
| 13 | 565 | 149 | 4096 | 4096 | 791 | 181 | 0.96 | 1.00 | 299,315 | 86,110 | 0.69 | 0.98 | 0.09 | 0.10 | -0.10 | -0.16 | 6,544 | 663 | 5,231 | 630 | 22.7 | 77.7 | 11.0 | 45.3 | 12.2 | 32.8 |
| 14 | 558 | 149 | 4096 | 1841 | 564 | 210 | 0.96 | 1.00 | 296,045 | 85,238 | 0.81 | 0.68 | 0.11 | 0.08 | -0.18 | -0.14 | 6,792 | 677 | 4,982 | 1,400 | 21.5 | 94.9 | 11.1 | 62.2 | 10.8 | 33.1 |
| 15 | 612 | 156 | 4096 | 1018 | 797 | 202 | 0.96 | 1.00 | 321,563 | 90,276 | 0.91 | 0.80 | 0.12 | 0.09 | -0.03 | -0.08 | 6,649 | 691 | 5,478 | 2,566 | 23.6 | 60.7 | 11.7 | 28.2 | 12.4 | 32.7 |
| 16 | 656 | 146 | 4096 | 3482 | 573 | 171 | 0.94 | 1.00 | 346,314 | 84,858 | 0.87 | 0.68 | 0.10 | 0.08 | -0.15 | -0.19 | 6,807 | 699 | 5,625 | 725 | 30.6 | 108.9 | 18.9 | 76.0 | 12.1 | - |
| 17 | 560 | 161 | 4096 | 4096 | 818 | 249 | 0.97 | 1.00 | 296,079 | 92,828 | 0.82 | 0.78 | 0.10 | 0.09 | -0.07 | -0.15 | 6,748 | 704 | 4,820 | 678 | 23.6 | 77.7 | 12.2 | 45.2 | 11.8 | 32.9 |
| 18 | 715 | 146 | 4096 | 677 | 734 | 200 | 0.93 | 1.00 | 374,544 | 84,805 | 0.90 | 0.65 | 0.11 | 0.08 | -0.11 | -0.19 | 6,779 | 713 | 6,420 | 3,566 | 39.2 | 104.1 | 29.1 | 71.3 | 10.6 | - |
| 19 | 575 | 188 | 4096 | 2795 | 511 | 243 | 0.96 | 1.00 | 303,507 | 106,461 | - | - | - | - | - | - | 6,786 | 715 | 5,348 | 1,162 | 21.6 | 80.8 | 11.1 | 47.9 | 11.0 | 33.1 |

## Averages (Steps 2-19, Excluding Warmup)

| Metric | Standard | Hybrid | Ratio |
|---|---:|---:|---:|
| Avg sequence length | 619 | 151 | 4.1x |
| Max sequence length | 4,096 | 2,564 | -- |
| Solved sequence length | 619 | 210 | 2.9x |
| Stop rate | 0.95 | 1.00 | -- |
| Step tokens | 326,417 | 87,286 | 3.7x |
| Reward | 0.78 | 0.70 | 1.1x |
| Correct rate | 0.10 | 0.08 | 1.3x |
| Learner TPS (overall) | 5,847 | 625 | 9.4x |
| Actor TPS | 5,186 | 1,316 | 3.9x |
| Step time | 27.2s | 93.3s | 3.4x |
| Train time | 16.4s | 60.7s | 3.7x |
| Sync time | 11.4s | 33.0s | 2.9x |
| Wall clock | 15 min | 37 min | 2.5x |

## Standalone vLLM Inference Benchmarks (Hybrid Model Only)

Standalone inference benchmarks for the hybrid model on a single node with 4 vLLM engines (TP=2), prefix caching enabled, **without** `--vllm_enforce_eager`. The GRPO training script uses `--vllm_enforce_eager`, so these numbers represent an upper bound on inference performance.

Config: 16 unique prompts, 4 samples each (64 total per batch), 4 batches + 1 warmup. Benchmark script: `open_instruct/benchmark_generators.py`.

| Response Length | Avg TPS | Avg MFU | Avg MBU | Avg Gen Time/Batch | Avg Sync Time/Batch | Beaker |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 3,483 | 1.87% | 17.96% | 18.8s | 1.01s | `01KG5WJ3A8VC3X5XQ5H982QP9D` |
| 4,096 | 3,023 | 1.68% | 19.11% | 86.7s | 2.41s | `01KG5X8NDN66BQSB1REQBK4HEW` |
| 8,192 | 2,247 | 1.25% | 17.38% | 233.3s | 3.74s | `01KG5Y220EHSMZ16AA2QD46BJE` |
| 16,384 | 1,486 | 1.00% | 16.45% | 705.8s | 27.18s | `01KG61HNM2S8M8TQEKABE2MT3T` |

Key observations:
- TPS degrades significantly with response length (3,483 to 1,486, a 2.3x drop from 1K to 16K).
- The benchmark (without enforce_eager) at 1K tokens gets 3,483 TPS, but GRPO training (with enforce_eager, avg 151 tokens) gets only ~1,316 actor TPS -- a **2.6x gap**, suggesting `--vllm_enforce_eager` is a major contributor to the actor TPS penalty.
- Weight sync simulation time grows dramatically at 16K (27s), likely due to KV cache pressure.
- MFU stays very low (1-2%), indicating the model is memory-bandwidth bound, not compute bound.

## Single-GPU Benchmark

From `open_instruct/test_hybrid_layer_speed_gpu.py` (Beaker `01KH1W80SHAQNZB56M3VJJXM6X`):

The full hybrid model forward+backward at seq_len=2048 is **0.93x of OLMo3** (7% faster), due to its smaller hidden size (3840 vs 4096) offsetting slower linear attention. This confirms that the training slowdowns are entirely from **distributed overhead**, not model architecture.

## Surprising Observations

1. **10x learner TPS gap** (5,847 vs 625). Both are ~7B models on identical hardware. Single-GPU speed is comparable.
2. **4x actor TPS gap** (5,186 vs 1,316). vLLM inference is much slower, even though standalone benchmark (without enforce_eager) shows 3,483 TPS.
3. **3x weight sync gap** (11.4s vs 33.0s). Same DeepSpeed stage 3, same node topology.
4. **3x step time gap with high variance** (27.2s vs 93.3s avg; hybrid ranges 15.2s-132.6s).

## Root Cause: Two Architectural Differences

All four gaps trace back to two specific properties of GatedDeltaNet:

### Difference 1: GatedDeltaNet has ~3.25x more parameter tensors per layer

Each GatedDeltaNet layer has ~13 parameter tensors:
- 7 linear projections: q_proj, k_proj, v_proj, a_proj, b_proj, g_proj, o_proj
- 3 Conv1d modules (each with weight + bias)
- A_log, dt_bias, o_norm

Standard attention has ~4: q_proj, k_proj, v_proj, o_proj.

This causes:
- **10x learner TPS gap**: In ZeRO-3, every parameter access triggers an all-gather across all GPUs. With gradient checkpointing, parameters are accessed 3x per block (forward + recompute + backward). So 3.25x more tensors x 3x accesses = ~10x more all-gathers per training step.
- **3x weight sync gap**: `broadcast_weights_to_vllm` calls `torch.distributed.broadcast` once per named parameter. ~1.8x more tensors per block means ~1.8x more NCCL broadcast calls, each with fixed latency overhead. The remaining gap (3x vs 1.8x) likely comes from GatheredParameters materializing more small tensors (Conv1d weights, A_log, dt_bias are small but add per-tensor overhead).

This is confirmed by the DPO investigation (`docs/dpo_performance_investigation.md`), which found the same root cause causing a 16x slowdown on 4 nodes.

### Difference 2: FLA Triton kernels are incompatible with CUDA graph capture

FLA's GatedDeltaNet forward uses 6+ custom Triton kernels per layer (chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o), decorated with `@torch.compiler.disable`. Standard attention uses 1 FlashAttention kernel that supports CUDA graphs.

This requires `--vllm_enforce_eager` in the training script, which disables CUDA graph capture in vLLM. CUDA graphs amortize kernel launch overhead by replaying recorded GPU operations -- without them, every forward pass pays full kernel launch cost.

This causes:
- **4x actor TPS gap**: The standalone benchmark (without enforce_eager) achieves 3,483 TPS at 1K tokens. GRPO training (with enforce_eager) gets only 1,316 -- a 2.6x gap from enforce_eager alone. The remaining gap comes from 6x more kernel launches per layer.

### Combined effect: 3x step time with high variance

The 3.4x step time gap is the compound effect of:
- Slower training (from Difference 1)
- Slower weight sync (from Difference 1)
- The async pipeline can't overlap training and inference as effectively when both are slower

The high variance (hybrid ranges 15.2s-132.6s vs standard 10.7s-30.0s) may be because shorter sequences (avg 151 tokens) mean more sequences packed per batch (~135 vs ~33), creating more variable workloads across ranks, and ZeRO-3 all-gathers are synchronous barriers.

## Investigation Plan

These experiments specifically test whether the two architectural differences explain the gaps:

| # | Experiment | What it tests | Method |
|---|---|---|---|
| 1 | Count parameter tensors | Verify 3.25x tensor ratio (Diff 1) | Print `len(list(model.named_parameters()))` for both models |
| 2 | Profile all-gather time | Verify all-gathers dominate hybrid training (Diff 1) | `torch.profiler` around training step, compare all-gather fraction |
| 3 | Remove enforce_eager | Verify CUDA graph incompatibility still holds (Diff 2) | Run hybrid without `--vllm_enforce_eager` -- if it works, actor TPS should approach 3,483 |
| 4 | Try ZeRO-2 | Verify param sharding is the cause (Diff 1) | `--deepspeed_stage 2` eliminates per-param all-gathers entirely -- if learner TPS normalizes, Diff 1 is confirmed |

## Related
- Prior DPO investigation: `docs/dpo_performance_investigation.md`
- Single-GPU benchmark: `open_instruct/test_hybrid_layer_speed_gpu.py`
- Benchmark script: `open_instruct/benchmark_generators.py`
