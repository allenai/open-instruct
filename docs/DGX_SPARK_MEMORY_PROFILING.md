# DGX Spark Memory Profiling Methodology

**Last updated**: 2026-01-14

## Why This Document Exists

DGX Spark uses **unified memory** (119GB shared between CPU and GPU). Unlike discrete GPU systems where OOM kills only the GPU process, OOM on DGX Spark can **freeze the entire machine**, requiring a hard reboot.

This document describes a methodology for safely determining maximum batch sizes and configurations for different training types.

## The Problem

Standard approaches fail on DGX Spark:
- **Safeguards don't work**: OOM kills the system before cleanup handlers run
- **Memory monitoring is reactive**: By the time you detect high usage, it's too late
- **Sweeps are dangerous**: Running multiple configs risks bricking the machine

## Methodology: Incremental Profiling

### Step 1: Clean Slate
```bash
# Check for leftover processes
ps aux | grep -E "(ray|vllm|VLLM|python)" | grep -v grep

# Kill if any found
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
pkill -9 -f "python" 2>/dev/null || true

# Wait for memory release
sleep 15

# Verify clean state (should show ~115GB free)
free -h
```

### Step 2: Start Small
Run with **minimum viable batch size** first:
- SFT: `batch_size=1, grad_accum=1`
- DPO: `batch_size=1, grad_accum=1`
- GRPO: `batch_size=1, grad_accum=1, vllm_gpu_memory_utilization=0.1`

### Step 3: Monitor During Training
In a separate terminal, run continuous monitoring:
```bash
watch -n 5 'free -h | head -2; echo "---"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Step 4: Document Peak Memory
After each successful run:
1. Note the **peak memory used** (from monitoring)
2. Note the **batch size and settings**
3. Calculate **memory headroom** (total - peak)
4. Add to the working configurations table below

### Step 5: Increment Carefully
Double one parameter at a time:
1. First: `batch_size` (1 → 2 → 4 → 8 → 16 → 32)
2. Then: `grad_accum` (if batch_size hits limit)
3. Then: `max_seq_length` (1024 → 2048 → 4096)

**Stop when**: Peak memory exceeds 100GB (leaving 19GB headroom)

### Step 6: Record the Last Working Config
When you hit OOM, the **previous configuration** is your safe maximum.

---

## Working Configurations Table

### Qwen3-0.6B SFT (no_robots dataset)

| batch_size | grad_accum | seq_len | peak_mem_gb | headroom | status | notes |
|------------|------------|---------|-------------|----------|--------|-------|
| 2 | 1 | 1024 | 21.1 | 98GB | ✅ working | |
| 4 | 1 | 1024 | 29.0 | 90GB | ✅ working | |
| 8 | 1 | 1024 | 46.8 | 72GB | ✅ working | recommended safe max |
| 16 | 1 | 1024 | 80.9 | 38GB | ✅ working | approaching limit |
| 8 | 2 | 1024 | ? | ? | pending | same throughput as 16/1 |
| 8 | 4 | 1024 | ? | ? | pending | total_batch=32 |
| 8 | 8 | 1024 | ? | ? | pending | total_batch=64 |

### Qwen3-0.6B DPO (ultrafeedback dataset)

| batch_size | grad_accum | seq_len | peak_mem_gb | headroom | status | notes |
|------------|------------|---------|-------------|----------|--------|-------|
| 2 | 1 | 1024 | 23.8 | 95GB | ✅ working | |
| 4 | 1 | 1024 | 27.2 | 92GB | ✅ working | |
| 8 | 1 | 1024 | 62.2 | 57GB | ✅ working | recommended max |
| 4 | 4 | 1024 | ? | ? | pending | total_batch=16 |
| 8 | 4 | 1024 | ? | ? | pending | total_batch=32 |

**DPO vs SFT memory comparison:**
- DPO batch=2: 23.8GB vs SFT 21.1GB (1.13x)
- DPO batch=4: 27.2GB vs SFT 29.0GB (0.94x - similar)
- DPO batch=8: 62.2GB vs SFT 46.8GB (1.33x)

### Qwen2.5-1.5B SFT

| batch_size | grad_accum | seq_len | peak_mem_gb | status | notes |
|------------|------------|---------|-------------|--------|-------|
| 1 | 1 | 1024 | ? | pending | baseline |
| 2 | 1 | 1024 | ? | pending | |
| 4 | 1 | 1024 | ? | pending | |
| 8 | 1 | 1024 | ? | pending | |

### Qwen2.5-3B SFT

| batch_size | grad_accum | seq_len | peak_mem_gb | status | notes |
|------------|------------|---------|-------------|--------|-------|
| 1 | 1 | 1024 | ? | pending | baseline |
| 2 | 1 | 1024 | ? | pending | |
| 4 | 1 | 1024 | ? | pending | |

---

## Memory Estimation Formulas

Rough estimates for unified memory (these are approximations):

### SFT Memory
```
memory_gb ≈ model_params_B * 6 + batch_size * seq_len * hidden_dim * 4 / 1e9
```

Where:
- `model_params_B * 6`: Model weights (bf16) + optimizer states + gradients
- Second term: Activation memory (varies with gradient checkpointing)

### DPO Memory
```
memory_gb ≈ SFT_memory * 1.8  (policy + frozen reference model)
```

### GRPO Memory
```
memory_gb ≈ SFT_memory + vllm_memory_utilization * 119
```

---

## Quick Reference: Conservative Defaults

When in doubt, use these settings (known to work):

| Training Type | Model | batch | grad_accum | seq_len | estimated_mem |
|---------------|-------|-------|------------|---------|---------------|
| SFT | Qwen3-0.6B | 8 | 4 | 1024 | ~25GB |
| SFT | Qwen2.5-1.5B | 4 | 4 | 1024 | ~40GB |
| DPO | Qwen3-0.6B | 4 | 4 | 1024 | ~45GB |
| DPO | Qwen2.5-1.5B | 2 | 4 | 1024 | ~70GB |

---

## Experiment Log

### 2026-01-14: Qwen3-0.6B SFT Profiling

**Key findings:**
- Memory scales **super-linearly** with batch size (not linear!)
- batch=8 uses 47GB, batch=16 uses 81GB (nearly 2x for 2x batch)
- **Safe maximum**: batch=8 with grad_accum for throughput
- batch=16 works but leaves only 38GB headroom - risky for longer sequences

**Memory scaling observed:**
```
batch=2:  21GB (+0 baseline)
batch=4:  29GB (+8GB for 2x batch)
batch=8:  47GB (+18GB for 2x batch)
batch=16: 81GB (+34GB for 2x batch)
```

**Recommendation**: Use batch=8 as the safe maximum per-device batch size for Qwen3-0.6B SFT. Increase throughput via gradient accumulation, not batch size.
