<!-- Recovered from a research pass that stalled before assembling its final
     report. Sections 0-4 are complete; the intended sections 5 (per-model notes)
     and 6 (warm-start procedure) were never written. Findings below are the
     agent's, with vLLM source references it cited; they have NOT yet been
     verified against our cluster. Treat as a prioritised hypothesis list. -->

# Minimising vLLM server startup time for very large MoE models on 8x B300

**Scope:** process launch -> `vllm ready`, vLLM 0.28.0, 8x NVIDIA B300 (sm_103), CUDA 13.x, weights on WEKAFS.
**Target models:** Qwen3.5-397B-A17B-FP8, Kimi-K2.6 (INT4 W4A16), DeepSeek-V3.2-Exp (MLA+DSA), GLM-5.2-FP8 (MLA, `GlmMoeDsaForCausalLM`).

Evidence is tagged throughout:

- **[D]** documented behaviour (vLLM docs) or **verified in the v0.28.0 source tree** (file:line given; I unpacked the `v0.28.0` tag and read it).
- **[R]** reported experience (GitHub issues/PRs, vendor blogs).
- **[I]** my inference / extrapolation — treat as a hypothesis to measure.

---

## 0. Executive summary — the six things that matter

1. **You are almost certainly JIT-compiling FlashInfer kernels from source on every job.** `uvx vllm` installs `flashinfer-python` only. `flashinfer-cubin` is deliberately excluded from vLLM's wheel dependencies (`setup.py:1330-1334`) and `flashinfer-jit-cache` is only installed in the official *Docker image* (`docker/Dockerfile:761-766`). Without them, FlashInfer compiles every kernel with `nvcc` at startup — which is exactly why you have to install a CUDA toolkit in-job and why you see 40–55 min. Installing both wheels turns nearly all of that into a file read. **This is the single highest-value change.**
2. **Persist `VLLM_CACHE_ROOT` on weka, but also persist `TRITON_CACHE_DIR` yourself** — vLLM redirects Triton's cache into `VLLM_CACHE_ROOT` only on the *non-AOT* compile path (`compiler_interface.py:475-481`); on the AOT path (the default with torch >= 2.10, i.e. yours) it sets `TORCHINDUCTOR_CACHE_DIR` only (`decorators.py:550-559`). Triton then falls back to `~/.triton/cache` inside the container and is lost every job.
3. **Turn on `VLLM_ENABLE_STARTUP_PLAN=1`** (`envs.py:1876-1885`, `v1/worker/startup_plan.py`). It persists the memory-profiling result and skips both the profiling measurement and the CUDA-graph memory estimation pass on later boots with a matching fingerprint. Free, safe, off by default.
4. **Every vLLM env var you set is part of the torch.compile cache key** unless it is on an explicit ignore list (`envs.py:compile_factors()`, ~line 2215+). Sweeping `VLLM_*` knobs silently invalidates the compile cache. So does changing TP, DCP, `--kv-cache-dtype`, `--max-model-len` or `--max-num-batched-tokens`.
5. **CUDA graphs cannot be persisted** — they are captured into the live CUDA context. On data-centre Blackwell the default `max_cudagraph_capture_size` is **1024**, giving ~83 shapes, and the default mode `FULL_AND_PIECEWISE` captures *two* sets. Cutting the capture list is the only lever, and it is a pure throughput trade.
6. **Weight-load tail variance is a known lazy-mmap random-read pathology** (vLLM issue #40988: 3 of 8 ranks hung >60 min on a 1.6T checkpoint; fixed by `--safetensors-load-strategy prefetch`). You already use prefetch — the remaining lever is **prefetch parallelism**: `--safetensors-prefetch-num-threads` (default 8) and `--safetensors-prefetch-block-size` (default 16 MiB), `config/load.py:12-13`.

---

## 1. Compilation caching: what exists, where it lives, what controls it

### 1.1 The two compile paths in 0.28.0

vLLM 0.28 pins `torch==2.13.0` (`requirements/cuda.txt:7`). That matters:

```python
# vllm/envs.py:357-367
def use_aot_compile() -> bool:
    default_value = ("1" if is_torch_equal_or_newer("2.10.0")
                     and not disable_compile_cache() else "0")
    return os.environ.get("VLLM_USE_AOT_COMPILE", default_value) == "1"

# vllm/envs.py:369-377
def use_mega_aot_artifact():
    default_value = ("1" if is_torch_equal_or_newer("2.12.0.dev")
                     and use_aot_compile() else "0")
    return os.environ.get("VLLM_USE_MEGA_AOT_ARTIFACT", default_value) == "1"
```

**[D]** So on your stack **both `VLLM_USE_AOT_COMPILE` and `VLLM_USE_MEGA_AOT_ARTIFACT` default to 1.** This is important because the AOT path caches the *whole Dynamo-compiled callable*, not just Inductor output:

- **AOT path** (`vllm/compilation/decorators.py:525-575`): cache dir
  `$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/{hash}/`, with
  `{hash}/inductor_cache/` and `{hash}/rank_{rank}_{dp}/model`.
  A hit calls `torch.compiler.load_compiled_function(...)` and **skips Dynamo tracing entirely**. Log line on success: `Directly load AOT compilation from path ...` (`decorators.py:311-313`).
- **Legacy path** (`vllm/compilation/backends.py:1058-1105`): cache dir
  `$VLLM_CACHE_ROOT/torch_compile_cache/{hash_key}/rank_{rank}_{dp}/{prefix}/`, plus
  `inductor_cache/` and `triton_cache/` siblings created by
  `InductorAdaptor.initialize_cache` (`compiler_interface.py:462-481`).

This matters for your 1T-parameter models: the "Dynamo bytecode transform time" phase (which #20264 measured at 7.5 s on a tiny model, and which scales with layer count) is **only** eliminated on the AOT path. Do not set `VLLM_DISABLE_COMPILE_CACHE=1` — it also disables AOT compile.

### 1.2 The cache key — exactly what invalidates

`VllmBackend.configure_post_pass` / `aot_compile_hash_factors` build the key from four groups:

| Factor | Source | Notes |
|---|---|---|
| `env_hash` | `envs.compile_factors()` (`envs.py:2215+`) | **Every** `VLLM_*` env var except an explicit ignore list |
| `config_hash` | `VllmConfig.compute_hash()` (`config/vllm.py:423-529`) | model/cache/parallel/scheduler/attention/kernel/compilation/speculative configs |
| `code_hash` | SHA-256 over the *paths and contents* of all Dynamo-traced Python files (`compilation/caching.py:_compute_code_hash_with_content`) | legacy path only; AOT path re-verifies content at load (`decorators.py:265-281`) |
| `compiler_hash` | `get_inductor_factors()` — torch version, CUDA version, device capability | changing the wheel invalidates everything |

**[D] `compile_factors()` hashes every known vLLM env var and only excludes these** (abridged, from `envs.py`): `MAX_JOBS`, `VLLM_RPC_BASE_PATH`, `VLLM_USE_MODELSCOPE`, `VLLM_PORT`, **`VLLM_CACHE_ROOT`**, **`VLLM_ENABLE_STARTUP_PLAN`**, `VLLM_XLA_CACHE_PATH`, `VLLM_CONFIG_ROOT`, `LD_LIBRARY_PATH`, DP master ip/port, **`VLLM_FORCE_AOT_LOAD`**, S3 creds, usage-stats and all logging vars, `VLLM_TUNED_CONFIG_FOLDER`, **`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`**, `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`, all timeouts, all media/mm-cache vars, `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_ENABLE_V1_MULTIPROCESSING`, `LOCAL_RANK`, `CUDA_VISIBLE_DEVICES`, `NO_COLOR`.

Consequence **[D]**: `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM`, `VLLM_ATTENTION_BACKEND`, `VLLM_USE_FLASHINFER_MOE_INT4`, `VLLM_DEEP_GEMM_WARMUP`, `VLLM_HAS_FLASHINFER_CUBIN`, `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`, ... **all change the compile-cache directory.** Fix the whole `VLLM_*` environment across your sweep, or accept a cold compile per variant.

Note the good news: `VLLM_CACHE_ROOT` itself and `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` are explicitly ignored, so relocating the cache to weka does **not** invalidate it (the code comments say this is deliberate).

### 1.3 Answering question 2 directly

| Change | Invalidates torch.compile cache? | Why (source) |
|---|---|---|
| `--tensor-parallel-size` | **Yes** | `ParallelConfig.compute_hash` hashes all fields except an ignore list; `tensor_parallel_size` is not ignored (`config/parallel.py`). Also the artifact dir is per-rank (`rank_{rank}_{dp}`), so a TP change changes both the key and the rank layout. |
| `--decode-context-parallel-size` | **Yes** | `decode_context_parallel_size` is a `ParallelConfig` field (`parallel.py:342`) and is not in the ignore list. |
| `--prefill-context-parallel-size` | **Yes** | same, `parallel.py:126`. |
| `--enable-expert-parallel` | **Yes** | `ParallelConfig` field, not ignored. |
| `--kv-cache-dtype` | **Yes** | `CacheConfig.cache_dtype`; `CacheConfig.compute_hash` ignores only `gpu_memory_utilization`, `kv_cache_memory_bytes`, `num_gpu_blocks*`, `enable_prefix_caching`, block-size resolution helpers and `kv_sharing_fast_prefill`. |
| `--max-model-len` | **Yes** | `ModelConfig.compute_hash` ignore list (`config/model.py`) does **not** contain `max_model_len`. |
| `--max-num-batched-tokens` | **Yes** | `SchedulerConfig.compute_hash` hashes exactly this one field, explicitly because Inductor picks 32- vs 64-bit indexing from it (see vLLM issue #29585 cited in the code). |
| `--gpu-memory-utilization` | No | explicitly ignored in `CacheConfig.compute_hash`. |
| `--kv-cache-memory-bytes` | No | explicitly ignored. |
| `--enforce-eager` | No | `enforce_eager` is in `ModelConfig`'s ignore list — but it changes *whether* you compile at all. |
| `--load-format`, `--safetensors-load-strategy` | **No** | `LoadConfig.compute_hash` returns a constant (`config/load.py`): "this config will not affect the computation graph". Sweep loaders freely. |
| `--kernel-config.enable_flashinfer_autotune` / `enable_jit_warmup` / `enable_cutedsl_warmup` | **No** | explicitly ignored in `KernelConfig.compute_hash` (`config/kernel.py:262-267`). |
| `--cudagraph-mode`, `cudagraph_capture_sizes`, `max_cudagraph_capture_size` | **Yes** (they are `CompilationConfig` fields and `CompilationConfig.compute_hash` hashes *all* fields) | `config/compilation.py`; **[I]** in practice the Inductor/FX artifacts underneath are still reused via the nested `inductor_cache/`, so the miss is much cheaper than a true cold compile, but the top-level AOT artifact is re-created. |
| `--quantization`, model revision, `--dtype` | Yes | `ModelConfig` fields. |
| Different vLLM wheel / torch build / GPU model | Yes | `vllm.__version__` is an explicit factor; `get_inductor_factors()` includes torch + device capability. |

**Verification flag:** set `VLLM_FORCE_AOT_LOAD=1` and the boot *fails loudly* instead of silently recompiling (`decorators.py:326-327`; documented in `docs/configuration/optimization.md:23`). Use this in a canary job to prove your sweep dimensions do not bust the cache.

### 1.4 Can it be pre-built and shipped? Yes.

**[D]** The compile artifacts depend on the model *architecture and config*, not on weight values, and the docs state the directory "can be copied between machines or baked into a container image". The AOT artifact key (`decorators.py:255-262`, `_model_hash_key`) uses `vllm.__version__ + fn.__qualname__ + co_firstlineno` — **no absolute paths** — and `_verify_source_unchanged` compares content hashes, so relocating your install does not by itself invalidate the AOT cache. (The *legacy* path's `code_hash` does include absolute file paths, `backends.py:1040-1054` — one more reason to stay on the AOT path, and a reason to keep a stable install prefix anyway.)

Practical AOT warm-build recipe: run one throwaway `vllm serve` (or `vllm bench startup`) per (model, TP, dtype, max-model-len, max-num-batched-tokens, cudagraph config) tuple with `--load-format dummy` **[I]** — weights do not affect compilation, so a dummy-weight boot produces a valid compile cache at a fraction of the I/O. Verify with `VLLM_FORCE_AOT_LOAD=1` on the real boot.

> **[I] Caveat to measure:** `--load-format dummy` is a `LoadConfig` change, which is hash-neutral, so the cache *key* is right. What I have not verified is whether any quantization path specialises the compiled graph on values read from the checkpoint. For FP8/INT4 MoE the scales are parameters, not constants, so this should be safe — but prove it once with `VLLM_FORCE_AOT_LOAD=1`.

### 1.5 Inductor is single-threaded — by force

```python
# vllm/env_override.py:105
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
```

**[D]** This is an unconditional assignment (not `setdefault`), executed on `import vllm`, referencing vLLM issues #10480/#10619. You cannot raise it from the outside. It means cold Inductor compilation of a 60+-layer MoE is serial. **[I]** This is a large part of your cold-compile cost and reinforces that the only real answer is cache reuse, not more cores.

The same file sets `TRITON_CACHE_AUTOTUNING=1` (`env_override.py:113`), which writes Triton autotune results to `TRITON_CACHE_DIR` — see the gap in §1.6.

### 1.6 The Triton cache gap (actionable)

```python
# vllm/compilation/compiler_interface.py:475-481   (legacy path, cache enabled)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
os.environ["TRITON_CACHE_DIR"]        = triton_cache

# vllm/compilation/decorators.py:550-559           (AOT path — note what is missing)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
```

**[D]** On the AOT path — your default — `TRITON_CACHE_DIR` is never set by vLLM, and `caching.py:482-487` initialises the compiler manager with `disable_cache=True` against a `dummy_cache` dir, which returns *before* the `TRITON_CACHE_DIR` assignment. So Triton's own JIT cache (all the fused-MoE Triton kernels, the Mamba/GDN kernels, the sparse-MLA Triton kernels, plus autotune results) lands in the container-local default and dies with the job.

**Fix:** export `TRITON_CACHE_DIR=/weka/.../vllm-cache/triton` explicitly in the job. It is not a `VLLM_*` var, so it is not part of the compile hash.

---

## 2. Kernel JIT: the real 40–55 minutes

### 2.1 FlashInfer — three packages, only one of which you have

| Package | What it is | Where it comes from |
|---|---|---|
| `flashinfer-python` | core; **compiles/downloads kernels on first use** | PyPI |
| `flashinfer-cubin` | pre-compiled cubins for all supported archs (incl. Sm100a/Sm100f/**Sm103a**) | `https://flashinfer.ai/whl/` only — **not on PyPI since 0.6.14** |
| `flashinfer-jit-cache` | pre-built JIT modules for a specific CUDA version | `https://flashinfer.ai/whl/cu1XX` |

**[D]** vLLM 0.28 pins `flashinfer-python==0.6.16.post3` **and** `flashinfer-cubin==0.6.16.post3` in `requirements/cuda.txt:16-18`, but `setup.py:1330-1334` strips `flashinfer-cubin` from the published wheel's `install_requires`:

> "Not on PyPI since 0.6.14 (only https://flashinfer.ai/whl), so it cannot be a wheel dependency; flashinfer falls back to fetching cubins at runtime when the package is absent."

and `flashinfer-jit-cache` is installed **only** in `docker/Dockerfile:761-766`. **[I] A `uvx vllm==0.28.0` install therefore has neither.** That is the mechanism behind your 40–55 min: every trtllm-gen MoE / FMHA kernel is either fetched one-by-one from NVIDIA's artifactory (`FLASHINFER_CUBINS_REPOSITORY`, `vllm/utils/flashinfer.py:29-32`) or compiled from source with `nvcc`.

Corroborating **[D]** from the same file (`vllm/utils/flashinfer.py:36-63`):

```python
def has_flashinfer_cubin() -> bool:
    if envs.VLLM_HAS_FLASHINFER_CUBIN: return True
    if importlib.util.find_spec("flashinfer_cubin") is not None: return True
    ...
def has_flashinfer() -> bool:
    ...
    if not has_flashinfer_cubin() and shutil.which("nvcc") is None:
        # "FlashInfer unavailable since nvcc was not found and not using
        #  pre-downloaded cubins"
        return False
```

This is also why your image needs an in-job CUDA toolkit at all: without cubins, vLLM requires `nvcc` to consider FlashInfer usable.

**Fix (do this first):**

```bash
# match the index to torch's CUDA build, not the driver:
CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
uv pip install flashinfer-cubin==0.6.16.post3    --index-url https://flashinfer.ai/whl/
uv pip install flashinfer-jit-cache==0.6.16.post3 --index-url https://flashinfer.ai/whl/$CU
flashinfer show-config     # verify; also: flashinfer list-modules / module-status
```

The `flashinfer` CLI also exposes `install-cubin-wheel`, `install-jit-cache-wheel` and `download-cubin` for air-gapped pre-staging.

**FlashInfer cache dirs to persist** (from `flashinfer/jit/env.py`):

| Var | Default | Holds |
|---|---|---|
| `FLASHINFER_WORKSPACE_BASE` | `$HOME` | root of everything below |
| `FLASHINFER_CACHE_DIR` | `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer` | |
| `FLASHINFER_WORKSPACE_DIR` | `$FLASHINFER_CACHE_DIR/{version}/{arch}` | version+arch keyed, so safe to share |
| `FLASHINFER_JIT_DIR` | `$FLASHINFER_WORKSPACE_DIR/cached_ops` | **compiled JIT modules — persist this** |
| `FLASHINFER_GEN_SRC_DIR` | `$FLASHINFER_WORKSPACE_DIR/generated` | generated sources |
| `FLASHINFER_CUBIN_DIR` | package dir, else cache dir | downloaded cubins — persist if you cannot install the wheel |
| `FLASHINFER_AOT_DIR` | `flashinfer-jit-cache` package data | provided by the wheel |

Set `FLASHINFER_WORKSPACE_BASE=/weka/.../flashinfer-home` and the whole tree lands on weka. **[D]** vLLM keys its own FlashInfer autotune cache off `FLASHINFER_WORKSPACE_DIR`'s parent/name (`model_executor/warmup/flashinfer_autotune_cache.py:24-40`), so this stays consistent.

### 2.2 vLLM's warmup pipeline (what actually JITs, and how to switch pieces off)

`vllm/model_executor/warmup/kernel_warmup.py` is the orchestrator called from the worker. In order:

1. `warm_v1_block_table_kernels`, KV-block-zeroing kernel
2. `qwen_triton_warmup` — Qwen-specific Triton kernels
3. `deepseek_v4_mhc_warmup` — TileLang `hc_pre/hc_post/hc_head_op` (DSv4 only)
4. if `kernel_config.enable_jit_warmup` (default True): `kimi_k3_triton_warmup`, `fa4_cutedsl_warmup`, `sparse_mla_triton_warmup`
5. `_warmup_ll_bf16_router_gemm` on SM90+
6. if `kernel_config.enable_cutedsl_warmup` (default True): `cutedsl_warmup()`
7. `flashinfer_sparse_mla_decode_autotune_warmup`, `deepseek_v4_sparse_mla_attention_warmup`
8. **DeepGEMM warmup** if `VLLM_USE_DEEP_GEMM` and `VLLM_DEEP_GEMM_WARMUP != "skip"` (default `"relax"`)
9. `b12x_warmup`, `minimax_m3_msa_warmup`
10. **FlashInfer autotune** if `kernel_config.enable_flashinfer_autotune` and SM >= 90
11. FlashInfer attention warmup (`_dummy_run` with a mixed prefill/decode batch)

Knobs (all **hash-neutral**, `config/kernel.py:262-267`):

```bash
--kernel-config.enable_flashinfer_autotune=false   # skips step 10 entirely
--kernel-config.enable_jit_warmup=false            # skips step 4
--kernel-config.enable_cutedsl_warmup=false        # skips step 6
VLLM_DEEP_GEMM_WARMUP=skip|relax|full              # step 8; "relax" is default
VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm,...     # narrow step 10
```

**[D]** `VLLM_DEEP_GEMM_WARMUP` semantics from `envs.py:1570-1588`: `full` = every GEMM shape the engine could hit; `relax` = heuristic subset (default); `skip` = none. Setting `skip` moves the JIT cost to the first real request rather than removing it — good for a *latency-to-ready* metric, bad for first-token latency.

**Warning [D]:** `VLLM_DEEP_GEMM_WARMUP` and `VLLM_USE_DEEP_GEMM` are **not** in the compile-factor ignore list, so flipping them re-keys the torch.compile cache. `enable_flashinfer_autotune` / `enable_jit_warmup` / `enable_cutedsl_warmup` are safe to flip.

### 2.3 DeepGEMM JIT cache

```python
# vllm/utils/deep_gemm.py:260-266
DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
    os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(envs.VLLM_CACHE_ROOT, "deep_gemm")
```

**[D]** DeepGEMM's compiled kernels already land under `$VLLM_CACHE_ROOT/deep_gemm` by default, so pointing `VLLM_CACHE_ROOT` at weka persists them. You can also set `DG_JIT_CACHE_DIR` yourself (it is not a `VLLM_*` var, so hash-neutral).

### 2.4 FlashInfer autotune cache — persistable, with a live hazard

**[D]** `vllm/model_executor/warmup/flashinfer_autotune_cache.py` writes
`$VLLM_CACHE_ROOT/flashinfer_autotune_cache/{fi_version}/{arch}/{aot_hash}/autotune_configs.json`
(or under `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`). Rank 0 reads it, broadcasts the bytes to all ranks, every rank writes it locally, then `tuner.load_configs(...)`. Writes are atomic (`os.replace`).

**[R] Hazard:** vLLM PR #54618 (opened 2026-08-31, *after* the 0.28.0 release on 2026-08-26; closed unmerged 2026-09-09) reports that **loading a warm FlashInfer autotune cache under TP>1 with MoE models can deadlock at startup**: per-rank MoE problem shapes differ, so some ranks hit the cache and skip profiling while others miss and keep issuing per-tactic all-reduces, diverging the collective count. Rank 0 advances; the rest block. The author declined to land the "only load the cache when world_size == 1" workaround, pointing at PR #52292 (`VLLM_FLASHINFER_AUTOTUNE_DISTRIBUTED_SYNC=0`) — which is **still open and not in 0.28.0** (that env var does not exist in `vllm/envs.py` at v0.28.0).

**[I] Practical stance:** persist it, but know the symptom (hang immediately after `Using FlashInfer autotune cache file: ...`, ranks stuck in NCCL). Fallbacks, in order of preference:
1. `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$JOB_LOCAL_TMP` — cold-tune each job, lockstep-safe, keeps the perf benefit (costs one tune pass).
2. `--kernel-config.enable_flashinfer_autotune=false` — fastest boot, some steady-state loss, no deadlock.

Both are compile-hash-neutral.



---

## 3. CUDA graph capture — cost, savings, and what to cut

### 3.1 What is captured by default on B300

**[D]** `config/compilation.py:692-707`:

> "If not specified, `max_cudagraph_capture_size` is capped at 512 by default, **or 1024 on data center Blackwell GPUs**. This avoids OOM in tight memory scenarios ... and limits capture of large graphs that increase startup time and memory usage."

and the generated list is

```
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))
```

So on your B300s the default is **83 capture sizes** (3 + 31 + 49). Default `cudagraph_mode` is `FULL_AND_PIECEWISE` (`docs/design/cuda_graphs.md`), which captures **two** descriptor sets — uniform-decode (FULL) and mixed prefill-decode (PIECEWISE) — i.e. up to ~166 captures, each preceded by a dummy forward pass of a 400B–1T MoE model. The loop is `gpu_model_runner.py:7006-7050`; progress is the tqdm bar `Capturing CUDA graphs (decode|mixed prefill-decode, FULL|PIECEWISE)` and the final log is `Graph capturing finished in %.0f secs, took %.2f GiB` (`:7047-7051`). The code comment optimistically says "This usually takes 5~20 seconds" — that is written for small models.

### 3.2 Are CUDA graphs persistable? No.

**[D]/[I]** A CUDA graph is an object in the live CUDA context, holding device pointers into the current memory pool. Nothing in vLLM serialises them and CUDA provides no portable serialisation. Capture is unavoidably per-process. The research literature agrees this is the un-cacheable phase — the open proposals are *lazy* capture (vLLM RFC #20098) and template materialisation ("Foundry", arXiv 2604.06664), neither of which is in 0.28.0.

**Corollary:** capture time is a floor you can only lower by capturing less.

### 3.3 The knobs, with costs

| Setting | Startup effect | Steady-state cost |
|---|---|---|
| `--enforce-eager` | Removes **both** torch.compile and all capture. Fastest possible boot. | Largest loss: no Inductor fusions, full per-op launch overhead. Worst for small-batch decode, which is exactly where MoE models are launch-bound. |
| `-O0` | `cudagraph_mode=NONE`, `mode=NONE`, all fusions off, `enable_flashinfer_autotune=False`. | Same class of loss as eager, slightly better (custom ops still available). |
| `-O1` | `cudagraph_mode=PIECEWISE`, compile on, autotune on, two fusions. Halves the capture set (one descriptor set instead of two) and drops the `-O2` extra compile ranges. | Loses full-decode graphs: attention stays eager during decode. Typically a few % of decode throughput on MoE. **[I]** |
| `-cc.cudagraph_mode=FULL_DECODE_ONLY` | Captures **only** the uniform-decode set — roughly halves capture count vs `FULL_AND_PIECEWISE`, and also saves the piecewise-graph memory. | Prefill/mixed batches run eager. Fine for decode-heavy or P/D-disaggregated serving; hurts prefill-heavy sweeps. |
| `-cc.max_cudagraph_capture_size=256` | List becomes `[1,2,4] + range(8,256,8) + [256]` = 35 sizes, ~58% fewer captures than the B300 default of 83. | Batches above 256 tokens fall back to eager/piecewise. If your `--max-num-seqs` is <= 256 this costs nothing. |
| `-cc.cudagraph_capture_sizes='[1,2,4,8,16,32,64,128,256]'` | 9 sizes — ~90% fewer captures. | Padding waste: a batch of 100 replays the 128 graph. For decode-latency work this is usually a good trade; measure. |
| `VLLM_ENABLE_CUDAGRAPH_GC=1` | **Slower.** Default 0 means vLLM calls `gc.freeze()` around capture (`gpu_model_runner.py:6663-6677`). Leave at 0. | — |

**[I] Recommended sweep default:** keep `-O2`/`FULL_AND_PIECEWISE` for the runs whose numbers you will publish, but for *configuration-search* runs use
`-cc.cudagraph_mode=FULL_DECODE_ONLY -cc.max_cudagraph_capture_size=256`,
which should cut capture work by roughly 4x (half the descriptor sets x 35/83 of the sizes) at a throughput cost confined to prefill-heavy batches. Note this **does** change the compile hash, so keep one pinned value per cache lineage rather than sweeping it.

### 3.4 Skip memory profiling entirely

Two mechanisms, both worth using:

- **[D] `VLLM_ENABLE_STARTUP_PLAN=1`** (`envs.py:1876-1885`; implementation `v1/worker/startup_plan.py`). Each worker persists the profiled KV-cache-memory value and the free-memory baseline to `$VLLM_CACHE_ROOT/startup_plan/startup_plan_{fingerprint}.json`. Later boots auto-apply it — **skipping the memory-profiling measurement and the CUDA-graph memory estimation pass** — iff the fingerprint matches and current free memory >= the recorded baseline. The fingerprint is `VllmConfig.compute_hash()` + device name + total memory + compute capability + torch/CUDA version + rank + world size (`startup_plan.py:40-74`). It is opt-in, hash-neutral (explicitly in the ignore list), and a stale plan is simply ignored — "a stale plan costs nothing and is never trusted".
- **[D] `--kv-cache-memory-bytes <N>`** (the CLI spelling in 0.28 is `--kv-cache-memory-bytes`, `engine/arg_utils.py:1211`; the docs and log messages say `--kv-cache-memory`). vLLM logs the exact value on a successful boot; passing it back skips the same two passes. Risk noted in `docs/configuration/optimization.md:24`: too low caps concurrency, too high fails at allocation, and the value is only valid on the same GPU with the same initial free memory.

The startup plan is strictly better for a sweep because it is keyed and self-invalidating. Use `VLLM_ENABLE_STARTUP_PLAN=1` and let it manage the value.

---

## 4. Weight loading from WEKA

### 4.1 What `--safetensors-load-strategy` actually does

**[D]** `config/load.py:62-93`, implementation `model_executor/model_loader/weight_utils.py:828-960`:

| Value | Behaviour |
|---|---|
| unset (`None`) | lazy mmap, **plus** auto-prefetch iff the FS type is recognised as network *and* the checkpoint fits in 90% of available RAM |
| `lazy` | mmap only; explicitly suppresses auto-prefetch |
| `eager` | `load(open(f,'rb').read())` — whole file read sequentially into RAM, then parsed. **No random reads at all.** Highest RAM use. |
| `prefetch` | background threads stream every file through the page cache while the normal lazy load proceeds |
| `torchao` | torchao subclass reconstruction |

**Critical for you [D]:** the network-FS detection is

```python
fs_type = _get_fs_type(sorted_files)          # reads /proc/mounts, longest-prefix match
is_net_fs = fs_type in ("nfs", "nfs4", "lustre")
```

`wekafs` is **not** in that list, so vLLM will log *"Auto-prefetch is disabled because the filesystem (WEKAFS) is not a recognized network FS (NFS/Lustre)"* and do nothing unless you pass the flag. You already pass it — good, and this explains why it was necessary.

### 4.2 The prefetch is rank-sharded — and tunable

```python
# weight_utils.py:754-826
paths_to_prefetch = sorted_files[rank::world_size]     # each rank takes 1/world_size of the files
with ThreadPoolExecutor(max_workers=num_prefetch_threads) as executor: ...
threading.Thread(target=_run_prefetch, daemon=True).start()   # background, load proceeds concurrently
```

**[D]** Defaults are `DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS = 8` and `DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE = 16 MiB` (`config/load.py:12-13`). With TP=8 that is 8 ranks x 8 threads = **64 concurrent 16 MiB sequential streams** against WEKA. Your measured ~5 min for a 400–750 GB checkpoint implies ~1.5–2.5 GB/s aggregate, which is well under what a WEKA cluster will give 64+ streams.

**Actionable:**

```bash
--safetensors-load-strategy=prefetch \
--safetensors-prefetch-num-threads=32 \      # -> 256 concurrent streams on a TP=8 node
--safetensors-prefetch-block-size=67108864   # 64 MiB
```

**[I]** Expect the prefetch phase to drop roughly linearly with achieved bandwidth until you saturate the node's network or WEKA client; 5 min -> 1–2 min is a reasonable target. Tune by watching the log lines `Prefetching checkpoint files: N% (a/b)` and `Prefetching checkpoint files into page cache finished in %.2fs`.

### 4.3 Why one node took 57 minutes for 27% of shards

**[R]** vLLM issue #40988 is the same failure mode, on DeepSeek-V4-Pro (1.6T, 805 GiB checkpoint, ~102 GiB/rank, TP=8, EXT4/NVMe): 5 of 8 workers finished in 6–9 minutes, **3 workers hung >60 minutes** doing random reads through `safetensors._safetensors_rust.safe_open`. Root cause: lazy mmap issues ~13,000 random reads per rank scattered across the shard during weight registration (quantization-config fixup), and readahead does not cover the MoE expert layout. Fix: `--safetensors-load-strategy prefetch`, which turns it into one sequential bandwidth-bound pass; cold start then completed in ~12 min. V4-Flash at ~37 GiB/rank did not hang, so the threshold sits between 37 and 102 GiB/rank.

**[I] Applied to you:** prefetch is a *background* thread racing the loader. On a slow node the loader overtakes the prefetcher and you are back to random reads — which is exactly a 27%-of-shards, 8x-variance signature. Two hardening options:

1. **Pre-warm the page cache before vLLM starts**, synchronously, in the Beaker job script. Then the loader never faults to disk:
   ```bash
   find "$MODEL_DIR" -name '*.safetensors' -print0 \
     | xargs -0 -P 32 -I{} dd if={} of=/dev/null bs=64M status=none
   ```
   This also gives you a clean, separately-timed phase you can alert on, and lets you fail fast on a slow node before burning GPU-minutes.
2. **`--safetensors-load-strategy=eager`** — reads each file whole, sequentially, before parsing. Removes random reads by construction. Costs transient RAM of one file per rank (typically ~5 GB x 8 ranks) on top of the page cache. **[I]** This is the most deterministic option if you have the RAM; it is the strategy the docs recommend "for models on network filesystems (e.g. Lustre, NFS) ... it avoids inefficient random reads".

### 4.4 Expert-parallel weight filtering

**[D]** `--enable-ep-weight-filter` (added by PR #37351, merged 2026-03-18). Implementation `model_executor/model_loader/ep_weight_filter.py` + `default_loader.py:351-400`. Each rank computes its local expert ids and skips non-local expert tensors **before reading them from disk**. The module docstring: "experts typically account for ~85-90 % of total weight bytes".

Preconditions (all enforced in `_init_ep_weight_filter`):
- `model_config.is_moe` **and** `parallel_config.enable_expert_parallel` **and** `enable_ep_weight_filter`
- **not** `--enable-eplb` (redundant expert slots need all logical experts)
- checkpoint must use **per-expert** tensor names matching `\.experts\.(\d+)\.` — 3D fused-expert checkpoints (GPT-OSS style) are unaffected

So: DeepSeek-V3.2, Kimi-K2.x and GLM-MoE layouts qualify; verify per checkpoint with `python -c "import json;print([k for k in json.load(open('model.safetensors.index.json'))['weight_map'] if '.experts.' in k][:3])"`.

**[I] Important interaction:** `--enable-ep-weight-filter` reduces the bytes each rank *parses and copies*, but `--safetensors-load-strategy=prefetch` still streams **every** file into the page cache (`_prefetch_all_checkpoints(sorted_files, ...)` is not filtered). On a single TP=8 node with a shared page cache that is the right behaviour anyway — the node must read each byte once regardless. The win from the EP filter is therefore mostly CPU/memcpy and per-rank RSS, not network bytes. It becomes a *large* win if the checkpoint does not fit in RAM (then filtered ranks avoid re-faulting evicted pages).

### 4.5 Other loaders

| `--load-format` | Notes |
|---|---|
| `instanttensor` | **[D]** New in recent vLLM (`config/load.py:39-41`, `weight_utils.py:1102-1145`, RFC #36091). Distributed loading over the existing `torch.distributed` world group, pipelined prefetch, direct I/O (bypasses the page cache), GDS / legacy / memory backends. vLLM's own docs report Qwen3-30B-A3B 57.4 s -> 1.77 s single-GPU (32.4x) and **DeepSeek-R1 on 8x H200 160 s -> 15.3 s (10.5x)**. Recommended by its docs when storage bandwidth >= 5 GB/s or when you cannot keep the model in host RAM. `pip install instanttensor`, then `--load-format instanttensor`. **This is the most promising alternative to prefetch for your setup** and is hash-neutral so you can A/B it freely. **[I]** Direct I/O sidesteps the page-cache-fill cost entirely, which would collapse your 5 min prefetch + 7 min load into one overlapped phase. |
| `runai_streamer` | Concurrent reads; `--model-loader-extra-config '{"concurrency":N}'`. **[R]** Route179 measured 142 s -> 59 s (2.4x), ~1.2 GB/s -> ~10 GB/s. |
| `runai_streamer_sharded` / `sharded_state` | Pre-sharded checkpoints; each rank reads only its own file. **[I] TP-size-locked** — you would need one converted copy per TP degree, which fights your sweep. |
| `tensorizer` | Requires a serialisation pass; strong for S3, less compelling for a POSIX FS. |
| `fastsafetensors` | Already a vLLM dependency (`requirements/cuda.txt:26`). GDS-oriented. |
| multi-thread default loader | `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'`. **[D] Mutually exclusive with any non-lazy `safetensors_load_strategy`** — `default_loader.py:116-126` raises rather than silently dropping your strategy. |

### 4.6 Post-load weight processing — the phase nobody measures

**[D]** After the read, `model_executor/model_loader/utils.py:96-145` walks every module calling `quant_method.process_weights_after_loading(module)` — repacking, requantisation, Marlin permutation — **serially, on the GPU**, then `release_device_memory_under_pressure`. For an INT4 W4A16 MoE this is where per-expert Marlin repacking happens (`compressed_tensors_moe_wna16.py:500-562`: `marlin_moe_permute_scales`, `moe_packed_to_marlin_zero_points`, `replace_parameter` per expert group).

**[D]** The WNA16 backend oracle (`fused_moe/oracle/int_wna16.py:107-121`) prefers, in order: `FLASHINFER_TRTLLM` > `MARLIN` > `BATCHED_MARLIN` > `TRITON` > `HUMMING` > `EMULATION`. `FLASHINFER_TRTLLM` is rejected if the checkpoint has zero points or bias. **[I]** So a symmetric, zp-free INT4 checkpoint lands on the trtllm path (which needs cubins — see §2.1); anything with zero-points falls back to Marlin and pays the repack. This is the most likely explanation for a Kimi-class model showing a long, silent gap between "Loading safetensors checkpoint shards 100%" and the first compile log.

You can force the choice with `--kernel-config.moe_backend=<marlin|flashinfer_trtllm|triton|deep_gemm|...>` (`config/kernel.py`, `MoEBackend` literal) — but note `KernelConfig.compute_hash` hashes `moe_backend`, so it **does** re-key the compile cache.

### 4.7 The 8x node/rank variance has a named root cause — and it is not the node

**[R] vLLM issue #39030, "Certain Ranks Take a Long Time to Load Weights"** (opened 2026-04-05 by a vLLM maintainer, **still open**): on a B200 VM with 163 safetensors shards and 4 ranks, one worker took **197 s** while the other three finished in ~27–30 s — a 7x spread, i.e. exactly your signature. Diagnosed root cause: `safetensors_weights_iterator` iterates the file list in the *same* `_natural_sort_key` order on **every rank**, so all N ranks convoy on the same file simultaneously — storage-side contention plus page-cache thrash, and whichever rank starts late is starved.

**[D]** I confirmed this in your tree: `weight_utils.py:846` does `sorted_files = sorted(hf_weights_files, key=_natural_sort_key)` with no per-rank rotation, and `grep -c "rotate\|stagger" weight_utils.py` returns **0**.

**[R] The fix, PR #40068 ("Stagger checkpoint file reads across ranks"), is still OPEN** and not in 0.28.0. It is 13 lines: rotate each rank's file list by `offset = len(files) * rank // world_size`, applied *after* `_prefetch_all_checkpoints` so the prefetch sharding stays correct. It also flips the load-timing log from `scope="local"` to `scope="process"`.

**Two things to take from this:**

1. **Take the logging change regardless.** In 0.28.0 the weight-load timing is logged with `scope="local"` — **you only see local rank 0's number**. That is why your slow-shard problem looks like node variance: you cannot currently see per-rank times at all. Without this you are debugging blind.
2. **Sidestep the convoy.** `--safetensors-load-strategy=prefetch` already pulls files in a *rank-staggered* order into the page cache ahead of the convoy (`sorted_files[rank::world_size]`), which is why it helps so much. Cherry-picking #40068 removes the convoy in the parse loop too. `sharded_state` / `runai_streamer_sharded` remove it structurally (per-rank files).

### 4.8 `OMP_NUM_THREADS` oversubscription — up to 15x slower loading

**[R] vLLM issue #52330** (2026-08-14), and **[D]** I verified the code is unchanged in 0.28.0:

```python
# vllm/v1/executor/multiproc_executor.py:1090-1107
def set_multiprocessing_worker_envs(local_world_size: int = 1):
    _maybe_force_spawn()
    if current_platform.is_cpu() or "OMP_NUM_THREADS" in os.environ:
        return                                  # user-set value is respected untouched
    num_threads = startup_omp_num_threads(local_world_size)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
```

`local_world_size` carries **no data-parallel term** (`world_size = tp * pp * pcp`). At DP=4 on a 224-CPU node, four engine cores each claim 224 threads -> 896 threads on 224 CPUs. Single-variable A/B from the issue (Kimi-K2-Thinking-NVFP4, 4x B200, 224 CPUs, `runai_streamer`):

| `OMP_NUM_THREADS` | node-wide threads | weight load | outcome |
|---|---|---|---|
| unset (vLLM default) | 896 | **1822.4 s** | `TimeoutError` at 600 s |
| `56` | 224 | **118.3 s** | serves normally |

Dose-response: DP=1 -> 56 s; DP=4 with OMP=56 -> 118 s; DP=8 with OMP=56 -> 671 s; DP=4 default -> 1822 s; DP=8 default -> 2059 s. Reproducible to 0.38%. Bisected to PR #49919.

**Why this is aimed straight at you:** the reporter notes the slowdown scales with **tensor count, not bytes** — an NVFP4/INT4 checkpoint deserialises into **~277,000 small tensors** just above torch's `GRAIN_SIZE`, so nearly every tensor pays full thread-pool dispatch against an oversubscribed pool, whereas "equivalent bf16 checkpoints load as ~51 large tensors and are essentially unaffected." **Kimi-K2.6 INT4 and any NVFP4 variant are exactly the worst case.**

**Action:** set `OMP_NUM_THREADS` explicitly. For plain TP=8 on one node vLLM's own default is already right; the moment you add `-dp N` on the same node it is wrong by a factor of N. **[I]** This is also a plausible contributor to your "slow node" — a node with a different core count gets a different (and possibly much worse) default.

### 4.9 Raise the engine-ready timeout before you measure anything

**[D]** `VLLM_ENGINE_READY_TIMEOUT_S` defaults to **600 s** (`envs.py:27`, `:800`). Your measured startups run to 5360 s. Any run that trips this dies with `TimeoutError: Timed out waiting for engine core processes to start. This is often caused by slow weight loading for large models.` while the engine cores keep running. Set `VLLM_ENGINE_READY_TIMEOUT_S=3600` for the duration of this work. **[R]** This exact failure is what vLLM issue #32116 hit with a 19-minute DeepGEMM warmup.

Caution: it is **not** in the compile-factor ignore list, so changing it re-keys the compile cache. Pick one value and keep it fixed across the sweep.

### 4.10 Loader corrections for 0.28.0

- **`--load-format bitsandbytes` no longer exists.** It is `--quantization bitsandbytes` now.
- **`--load-format runai_streamer_sharded` maps to `ShardedStateLoader`, not the RunAI loader class.** `ShardedStateLoader.__init__` accepts only `{"pattern"}` in `--model-loader-extra-config` and raises on anything else — so the command in `docs/models/extensions/runai_model_streamer.md` (`--load-format runai_streamer_sharded --model-loader-extra-config '{"concurrency":16,...}'`) **errors out**. Set `RUNAI_STREAMER_CONCURRENCY` / `RUNAI_STREAMER_MEMORY_LIMIT` as plain env vars instead; the C++ streamer reads them directly.
- **`fastsafetensors` cannot use GPUDirect Storage at TP>1.** `weight_utils.py:1054-1057`: `nogds = pg.size() > 1`, unconditionally, with the comment that `cuFileDriverOpen()` would create CUDA contexts on every visible GPU. WekaFS *is* a GDS-qualified filesystem, but you cannot reach that path through vLLM 0.28 without patching that line. Only knob: `VLLM_FASTSAFETENSORS_QUEUE_SIZE`.
- **Sharded-state save script moved** to `examples/features/sharded_state/save_sharded_state_offline.py`. The dump is locked to TP size *and* to the quantization config *and* in practice to a compatible vLLM version (it stores the post-`process_weights_after_loading` state dict). No PP/DP rank in the filename pattern — treat it as TP-only.
- **`--max-parallel-loading-workers` is a throttle, not an accelerator** — and `parallel.py` logs `"max_parallel_loading_workers is currently not supported and will be ignored"` in 0.28 anyway.
- **`RUNAI_STREAMER_DIST` / `{"distributed": true}`: do not enable on a filesystem.** Measured 0.83x (20% *slower*) than plain `runai_streamer` on DeepSeek-R1/8xH200 in the InstantTensor benchmark; Run:ai's own docs scope distributed streaming to object storage, where the page cache cannot help.
- **`--numa-bind` / `--numa-bind-nodes` / `--numa-bind-cpus`** exist in 0.28 (`vllm/utils/numa_utils.py`) and are excluded from the compile hash. Note `--numa-bind` forces the `spawn` start method, costing a few seconds of interpreter startup. **[I]** Worth an A/B on a dual-socket B300 node, where cross-socket memory traffic during a 700 GB load is not free.

### 4.11 Published weight-loading numbers worth calibrating against

All **[R]**, all on different storage than yours — use them for shape, not for absolute prediction.

| Change | Model / hardware | Before -> after | Source |
|---|---|---|---|
| `--safetensors-load-strategy eager` | Lustre, GKE A3M | **94 min -> 14 min** | vLLM PR #24469 (the PR that introduced the flag) |
| `--safetensors-load-strategy prefetch` | DeepSeek-V3-0324, 8x B200, TP=8 | server 1 **1500 s -> 435 s**; server 2 **3660 s -> 215 s (17x)** | vLLM PR #36012 |
| `--safetensors-load-strategy prefetch` | DeepSeek-V4-Pro 1.6T, EXT4 | 3 ranks hung >60 min -> whole boot ~12 min | vLLM issue #40988 |
| `--enable-ep-weight-filter`, Kimi-K2.5-NVFP4 591 GB, 384 experts | EP=8, warm cache | 58.45 s -> 25.35 s (2.3x); EP=16: 54.84 s -> 21.71 s (2.5x); single-node EP=4: 96.58 s -> 67.59 s (1.4x) | vLLM PR #37136 |
| `--load-format instanttensor` | DeepSeek-R1, 8x H200, TP=8/EP=8, 50 GB/s NVMe | **160 s -> 15.3 s (10.5x, 45 GB/s)** | InstantTensor benchmark + vLLM docs |
| `--load-format fastsafetensors` (nogds path) | same bench | 160 s -> 75.8 s (2.11x) | same |
| `--load-format runai_streamer` | same bench | 160 s -> 101 s (1.58x) | same |
| `runai_streamer` **distributed** | same bench | 160 s -> **192 s (0.83x, slower)** | same |
| `--load-format runai_streamer` + `{"concurrency":8}` | ~30B, local NVMe | 142 s -> 59 s (2.4x) | Route179 |
| `OMP_NUM_THREADS=56` vs unset | Kimi-K2-Thinking-NVFP4, 4x B200, DP=4 | **1822 s -> 118 s (15x)** | vLLM issue #52330 |

**[I] The EP-filter numbers need one correction for your topology.** Those are *multi-node DP/EP* measurements. On a single TP=8 node with a shared page cache, the union of what your 8 ranks read is still the whole checkpoint, so the network-byte saving is small; what you actually save is per-rank `get_tensor()` work, copies and RSS — which is the 1.4x single-node figure, not the 2.5x one. The filter also does **not** apply to `runai_streamer`, `fastsafetensors`, `instanttensor`, `sharded_state`, or `enable_multithread_load` (`local_expert_ids` is only threaded into `safetensors_weights_iterator`). Look for the log line `EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts` to confirm it engaged.

---

## 5. Per-model startup notes

> **Provenance for this section.** Unless marked **[R]**, every code claim below was
> read out of an unpacked `v0.28.0` source tree (`github.com/vllm-project/vllm`,
> tag `v0.28.0`), and every model claim out of the live
> `huggingface.co/api/models/<repo>` listing and `raw/main/config.json`. Line
> numbers are from that tree. **[R]** items are GitHub issues/PRs — note that
> several are *open*, i.e. the bug is in the version you are running.

### 5.0 The four at a glance

| | Qwen3.5-397B-A17B-FP8 | Kimi-K2.6 | DeepSeek-V3.2-Exp | GLM-5.2-FP8 |
|---|---|---|---|---|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` | `KimiK25ForConditionalGeneration` | `DeepseekV32ForCausalLM` | `GlmMoeDsaForCausalLM` |
| `model_type` | `qwen3_5_moe` | `kimi_k25` | `deepseek_v32` | `glm_moe_dsa` |
| vLLM module | `models/qwen3_5.py` | `models/kimi_k25.py` | `models/deepseek_v2.py` | `models/deepseek_v2.py` |
| layers / routed experts | 60 / 512 (top-10) | 61 / 384 (top-8) | 61 / 256 (top-8) | 78 / 256 (top-8) |
| safetensors shards | **94** | **64** | **163** | **141** |
| quant | FP8 blockwise `[128,128]`, dynamic | compressed-tensors `pack-quantized` 4-bit, **group 32, symmetric** | FP8 `[128,128]`, **`scale_fmt: ue8m0`** | FP8 `[128,128]` e4m3 |
| MTP head in ckpt | yes (`mtp.fc`, `mtp.layers.0.*`) | **no** (`num_nextn_predict_layers: 0`) | yes (`=1`) | yes (`=1`) |
| sparse-attention indexer | no | no | yes (`index_topk: 2048`, `index_n_heads: 64`) | yes (`index_topk: 2048`) |
| `--trust-remote-code` needed? | **no** (no `.py` in repo) | **no** for config (vLLM bundles `KimiK25Config`); repo *does* ship 9 `.py` files + `auto_map` | **no** (the `.py` files are a standalone `inference/` reference impl, not `auto_map`) | **no** (no `.py` in repo) |
| `@support_torch_compile` | **yes**, `qwen3_5.py:206` | LM: no decorator found; ViT: bare `@torch.compile` (`kimi_k25_vit.py:64`) | yes, `deepseek_v2.py:1360` | yes, same decorator (subclass) |
| unique startup JIT | GDN Triton trio + **FlashInfer GDN prefill** + vision tower | Marlin repack *or* trtllm-gen INT4 cubins | DeepGEMM indexer + sparse-MLA Triton metadata + CuTe-DSL | same as DeepSeek **+ SM103-only CuTe-DSL skinny GEMMs** |

**[D] None of the four needs `--trust-remote-code` on 0.28.0.** vLLM ships bundled
config classes for all of them —
`transformers_utils/config.py:88` (`deepseek_v32="DeepseekV3Config"`),
`:104` (`kimi_k25="KimiK25Config"`),
`:135` (`qwen3_5_moe="Qwen3_5MoeConfig"`),
and GLM is handled by `_PATCH_HF_ALLOWED_LAYER_TYPES = {"glm_moe_dsa": ("deepseek_sparse_attention",)}`
(`transformers_utils/config.py:151-154`), which extends transformers' strict
`ALLOWED_LAYER_TYPES` so the checkpoint validates without remote code.
Our current recipe in `README.md` passes `--trust-remote-code` for DeepSeek, GLM
and Kimi. **[I]** Dropping it removes an HF hub round-trip and an `exec` of
downloaded Python on every rank at import; for Kimi it avoids `exec`-ing nine
files including a tiktoken-based `tokenization_kimi.py`. Drop it, keep
`HF_HUB_OFFLINE=1` set, and if something breaks it will break loudly at config
load, not 40 minutes in.

**[D] Two more redundancies in the current recipe.** `--tokenizer-mode deepseek_v32`
is now automatic: `config/model.py:683` sets `tokenizer_mode = "deepseek_v32"`
whenever `tokenizer_mode == "auto"` and the model calls for it (and logs
`Defaulting to tokenizer_mode=...`). And see §5.3 for why `VLLM_USE_DEEP_GEMM=0`
does **not** do what the recipe comment claims for the two DSA models.

---

### 5.1 Qwen/Qwen3.5-397B-A17B-FP8 — the only one that really pays for torch.compile

**What it is.** 60 layers, of which 15 are full attention (every 4th) and 45 are
Gated DeltaNet; 512 routed + 1 shared expert, top-10; a 27-block vision tower; an
MTP head living *inside* the main checkpoint (the FP8 `modules_to_not_convert`
list names `mtp.fc`, `mtp.layers.0.mlp.gate`, `mtp.layers.0.mlp.shared_expert_gate`);
94 safetensors shards, the fewest bytes of the four.

#### 5.1.1 It is the one model whose compile cache is worth warming

**[D]** `Qwen3_5Model` carries `@support_torch_compile(dynamic_arg_dims={...})`
at `vllm/model_executor/models/qwen3_5.py:206`. Everything in §1 — the AOT
artifact, `VLLM_FORCE_AOT_LOAD=1` as a canary, `--load-format dummy` warm-builds
— applies to Qwen with full force. The DSA pair (§5.3/§5.4) compile too but are
one config flip away from not compiling at all; Kimi's language model has no
decorator at all.

**[D]** `language_model_only` is a `ModelConfig.compute_hash` factor
(`config/model.py:457`), so `--language-model-only` — which our recipe passes —
**is part of the compile-cache key**. Fix it across the sweep or accept a cold
compile when you toggle it.

#### 5.1.2 GDN forces three Triton kernels and one FlashInfer JIT, and only three are warmed

**[D]** `vllm/model_executor/warmup/qwen_triton_warmup.py` is model-family-gated:

```python
_QWEN_MODEL_TYPES = frozenset(
    {"qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)   # L=1 constexpr, non-divisible L, divisible L
```

It compiles exactly three kernels, all of which the other three models never touch:

1. `causal_conv1d_fn` — `model_executor/layers/mamba/ops/causal_conv1d.py`
2. `fused_post_conv_prep` — vendored FLA, `vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py`, at **three** lengths (three Triton specialisations)
3. `fused_sigmoid_gating_delta_rule_update` — `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`

**[D] The chunked *prefill* GDN kernel is not in that list.** The prefill path is
selected separately by `_resolve_gdn_prefill_backend`
(`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:93-141`), and on a B300
it resolves to **FlashInfer**, not Triton:

```python
elif (current_platform.is_device_capability_family(100)   # SM10.x — B200/B300
      and head_k_dim == 128
      and current_platform.get_cuda_runtime_major() >= 13):
    supports_flashinfer = True
    supports_cutedsl = True
if backend in ["flashinfer", "auto"] and supports_flashinfer:
    return backend, "flashinfer"
```

Our stack satisfies all three conditions (`linear_key_head_dim: 128`, CUDA 13.1),
so the default `auto` gives us the FlashInfer GDN prefill kernel — **which is
JIT-compiled**. Worse, the warning that says so is gated on SM90:

```python
if active_backend == "flashinfer" and current_platform.is_device_capability(90):
    logger.warning_once("FlashInfer GDN prefill is JIT-compiled; first run may "
                        "take a while. Set --gdn-prefill-backend triton to skip JIT.")
```

**[D] On Blackwell you get the JIT and not the warning.** The only line you see is
`Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128).`

**[R]** vLLM PR #46764 ("[GDN] Improve UX when FlashInfer JIT compilation is
happening") logs the real-world shape of this:
`Warming up FlashInfer GDN prefill kernel (cold cache JIT-compile may take several minutes; cached under .../cached_ops)`
and recommends `--additional-config '{"gdn_prefill_backend":"triton"}'` for
"faster cold startup, potentially slower steady-state."

**Actions, in order:**

```bash
# (a) make the FlashInfer JIT a file read — this is §2.1's fix, and it is the
#     one that matters most for Qwen because the GDN prefill kernel is FlashInfer:
#     install flashinfer-cubin + flashinfer-jit-cache, persist FLASHINFER_JIT_DIR.
# (b) if you want a deterministic, JIT-free cold boot for config-search runs:
--additional-config '{"gdn_prefill_backend":"triton"}'
```

**[I]** `gdn_prefill_backend` lives in `additional_config`, which *is* hashed into
the compile key, so pin one value per cache lineage rather than sweeping it.
**[R]** Note also issue #56125: the vendored FLA `fused_recurrent_gated_delta_rule`
non-packed path is ~3x slower than upstream `flash-linear-attention` 0.5.2
(bit-identical), so the Triton fallback costs more steady-state than you would
guess from the upstream kernel.

#### 5.1.3 DeepGEMM is *auto-disabled* for this model on Blackwell

**[D]** `vllm/utils/deep_gemm.py:27-46`:

```python
_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {"qwen3_5_text", "qwen3_5_moe_text"}

def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+)."""
```

The text submodel of our checkpoint is `qwen3_5_moe_text`, and B300 is
`is_device_capability_family(100)`, so the E8M0 scale format is dropped back to
`FLOAT32`. **[I]** Consequence for startup: Qwen is the one model of the four
where our `VLLM_USE_DEEP_GEMM=0` is close to a no-op anyway — the FP8 MoE lands
on FlashInfer trtllm-gen or CUTLASS, not DeepGEMM. **[R]** This exclusion list
exists because of issue #47130 (DeepGEMM `"Unknown recipe"` assertion during FP8
kernel warmup on Blackwell for a `qwen3_5` FP8 checkpoint) — i.e. for this model
DeepGEMM was not slow, it *crashed* the warmup.

#### 5.1.4 The multimodal wrapper: two separate skips, and a way to prove they worked

**[D]** Startup runs a dummy forward through the vision tower during
`profile_run` unless you stop it. Three levers, all real in 0.28.0:

| Flag | Effect | Hash-neutral? |
|---|---|---|
| `--language-model-only` | vision tower not built at all (`config/multimodal.py:492`) | **No** — `ModelConfig.compute_hash` factor (`config/model.py:457`) |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | drives `mm_max_toks_per_item` to 0; `gpu_model_runner.py` then logs *"Skipping encoder profiling for embedding-only mode"* | **No** (MultiModalConfig field) |
| `--skip-mm-profiling` | skips the encoder profiling forward pass unconditionally (`config/multimodal.py:217`, used at `gpu_model_runner.py:6562`) | **No** — it is in `ModelConfig.compute_hash`'s init-var list (`config/model.py:445`) |

**[D] Verification signal.** Encoder compile time is tracked separately
(`compilation_config.encoder_compilation_time`, `config/compilation.py:750`) and
printed by the engine:

```
init engine (profile, create kv cache, warmup model) took %.2f s
    (compilation: %.2f s — language_model: %.2f s, encoder: %.2f s)
```
(`vllm/v1/engine/core.py:336-345`). **If `--language-model-only` really took
effect, the three-term form of this line disappears entirely** and you get the
`(compilation: %.2f s)` form. That is a one-grep regression test that our
text-only serving is actually text-only.

#### 5.1.5 MTP: a second model, a second capture set — and currently broken

**[D]** The MTP head is registered as its own architecture
(`registry.py:682-683`: `Qwen3_5MTP`/`Qwen3_5MoeMTP` → `qwen3_5_mtp`) and the
runner treats the drafter as a separate model: `gpu_model_runner.py:7339-7353`
initialises a *drafter* cudagraph dispatcher, and the capture loop calls the
drafter's dummy run for every shape in the target's capture list. The code
comment at `:2678` is explicit that "the drafter still only uses piecewise
cudagraphs." **[I]** So enabling MTP adds: one more weight-load pass, one more
Dynamo/Inductor compile, and one more piecewise capture set on top of the
target's `FULL_AND_PIECEWISE` — budget 1.3–1.5x the §3 capture time.

**[R] Do not enable it on 0.28.0 for this model.** Issue #55533 (open) and its
WIP fix #55617: *"Hybrid GDN (Qwen3.5/Qwen3.8 27B-class) + MTP: scheduler runs
only ~3 concurrent sequences at batch >= 4 — acceptance/throughput collapse."*
**[R]** #55369 (merged) was needed just to resolve `n_predict` from `text_config`
for Qwen3.5 *multimodal* MTP. **[R]** vLLM's own CI runs the
`qwen3_next_mtp_async_eplb` scheduled test with `VLLM_ENGINE_READY_TIMEOUT_S=1800`
— three times the default — which is the clearest available statement of how long
GDN+MTP cold start takes.

#### 5.1.6 Hybrid KV cache: constraints that bite at startup

- **[D]** `mamba_cache_mode == "all"` is rejected outright for this model:
  `qwen3_5.py:322-325` raises and tells you to use `--mamba-cache-mode=align`.
- **[R]** Issue #55766 (open, filed against v0.28.0): *"Qwen3.5/3.8 hybrid GDN:
  NaN logits after a prefix-cache hit when the previous prefill ended 4-10 tokens
  past a block boundary (mamba cache mode align)."* Our recipe passes
  `--enable-prefix-caching`. **[I]** Validate output correctness on this model
  before trusting a prefix-cached run; #51198/#51250 additionally report prefix
  caching being a silent 0%-hit no-op on this family, so you may be carrying the
  risk for no benefit.
- **[R]** Issue #37121: KV-cache capacity is over-estimated ~7x for hybrid
  Mamba/attention models because `unify_kv_cache_spec_page_size` pads the small
  Mamba state to the attention page size. **[I] This interacts directly with
  §3.4's startup plan**: `VLLM_ENABLE_STARTUP_PLAN=1` persists the *profiled*
  number, so it will faithfully persist the wrong one. It is still correct to use
  — it reproduces what a cold boot would have done — but do not read a persisted
  plan as validation of the number.
- **[R]** The vLLM recipes page for this family documents a hard failure,
  `cuda graph capture size is larger than mamba cache size`, whose fix is to lower
  `--max-cudagraph-capture-size` from the default. **[I]** That is the same knob
  §3.3 recommends for startup, so for Qwen the startup optimisation and the
  stability workaround are the same flag.

---

### 5.2 moonshotai/Kimi-K2.6 — the weight-processing model

**What it is.** `KimiK25ForConditionalGeneration` / `model_type: kimi_k25` — a
*multimodal* wrapper (27-layer MoonViT tower, patch-merger projector) around a
text backbone whose own `text_config` is `DeepseekV3ForCausalLM` / `kimi_k2`:
61 layers (1 dense + 60 MoE), 384 routed + 1 shared expert, top-8, MLA
(`kv_lora_rank 512`, `q_lora_rank 1536`, `qk_nope 128`, `qk_rope 64`, `v 128`,
64 heads). Moonshot's README says K2.6 "has the same architecture as Kimi-K2.5,
and the deployment method can be directly reused."

**[D] `num_nextn_predict_layers: 0` — it is the only one of the four with no MTP
head.** No drafter load, no second compile, no second capture set. Whatever
startup budget the others spend on speculative decoding, Kimi does not.

**[D] Checkpoint shape from `model.safetensors.index.json`:** `total_size`
595,148,192,736 B (~595 GB) across **64 shards**, and the weight map holds
**208,550 tensor entries**. That is the single most important number in this
subsection — see §4.8: the OMP-oversubscription pathology scales with *tensor
count*, not bytes, and 208 k tensors is the worst case in this fleet by an order
of magnitude. Set `OMP_NUM_THREADS` explicitly for this model even at DP=1.

#### 5.2.1 The INT4 backend actually selected is Marlin — because of a default-off env var

**[D]** `quantization_config` under `text_config`: `compressed-tensors`,
`format: pack-quantized`, `num_bits: 4`, `group_size: 32`, `strategy: group`,
`symmetric: true`, `dynamic: false`, and an `ignore` list of
`self_attn.*`, `shared_experts.*`, `mlp.(gate|up|gate_up|down)_proj.*`,
`lm_head.*`, `vision_tower.*`, `mm_projector.*` — i.e. **only the routed-expert
Linears are INT4**; everything else is BF16.

Symmetric + no bias means the WNA16 oracle's *only* disqualifier for
`FLASHINFER_TRTLLM` does not fire, and it is first in `_get_priority_backends()`
(`fused_moe/oracle/int_wna16.py`). **But it never gets a chance**:

```python
# vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py:24-31
def is_flashinfer_mxint4_moe_available() -> bool:
    return (envs.VLLM_USE_FLASHINFER_MOE_INT4        # <- envs.py:213, default False
            and has_flashinfer_trtllm_fused_moe()
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(100))
```

**[D] `VLLM_USE_FLASHINFER_MOE_INT4` defaults to `0` (`envs.py:213, 1624-1625`),
so out of the box Kimi-K2.6 lands on `MARLIN`, not trtllm-gen.** §4.6's warning
about a long silent gap after "Loading safetensors checkpoint shards 100%" is
therefore the *expected* behaviour for this model, not an edge case.

The checkpoint *is* format-compatible with the trtllm path — the kernel requires
`QuantKey(INT4, scale group_shape (1,32), symmetric=True)`, which is exactly
`group_size: 32, symmetric: true`. **[I]** So `VLLM_USE_FLASHINFER_MOE_INT4=1`
is worth one A/B, with two caveats: (a) it is a `VLLM_*` var and therefore
**re-keys the compile cache** (§1.2); (b) it is a default-off path for this
checkpoint shape, so validate output quality, not just speed.

#### 5.2.2 Why the Marlin repack is slow: it is a Python loop over 384 experts

**[D]** Three separate per-expert Python loops, each launching one tiny CUDA op
per expert:

```python
# vllm/_custom_ops.py — gptq_marlin_moe_repack (and awq_marlin_moe_repack)
for e in range(num_experts):
    output[e] = torch.ops._C.gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits, ...)
```
and `marlin_moe_permute_scales` / `moe_packed_to_marlin_zero_points` in
`quantization/utils/marlin_utils.py` are the same shape. `_process_weights_marlin`
in `fused_moe/oracle/int_wna16.py` calls the repack twice (w13, w2) and the
scale-permute twice per MoE layer.

**[D]** Because `symmetric: true`, `compressed_tensors_moe_wna16.py:517-520`
never registers zero-point parameters, so the zero-point loop is skipped.

**[I]** That leaves **4 per-expert loops x 384 experts x 60 MoE layers ≈ 92,000
launch-bound iterations per rank**, running concurrently across the 8 worker
processes but serially within each. This is launch-overhead-dominated, not
compute-dominated, which is why it does not get faster on a bigger GPU.

**[D] There is no on-disk cache of repacked weights.** Nothing in
`oracle/int_wna16.py`, `compressed_tensors_moe_wna16.py` or the loader persists
the post-`process_weights_after_loading` tensors. Every boot redoes it. **[I] The
only way to amortise it in 0.28.0 is `--load-format sharded_state`** (§4.5),
which dumps the *post*-processing state dict — at the cost of locking the dump to
one TP degree and one vLLM build. For a fixed production config that is a real
option; for a sweep it is not.

**[R] Measured anchor:** issue **#50968** (open) — Kimi-K2.6, 4x GB200 ARM64,
TP=4, vLLM 0.26.0: `Loading safetensors checkpoint shards: 100% | 64/64` then
**`Loading weights took 1228.97 seconds`** (~20.5 min) *before* the repack begins
— and then all four workers **segfault inside `gptq_marlin_repack`**
(`cuLibraryLoadData` → `cudaFuncSetAttribute` → `gptq_marlin_repack`), a
regression from 0.25.1 specific to ARM64/GB200. **[I]** If our nodes are x86 HGX
B300 this specific crash should not apply; if they are Grace-Blackwell GB300
(ARM64), test before trusting Marlin at all. The trace is also evidence that
Marlin's cubin is **lazily loaded into the CUDA context on first launch**, so
Marlin's "no JIT" property is about compilation, not about being free at startup.

**[R]** Same failure class historically: PRs #38669 / #46601 ("Fix Marlin repack
PTX incompatibility on H100/H200"). **[I]** Whether the published vLLM wheel ships
a native `sm_103a` cubin slice for Marlin or relies on PTX forward-compat from
`sm_100` is a build-configuration question we should answer by inspection of the
wheel (`cuobjdump --list-elf`) rather than assumption; our own `README.md` already
records one research pass concluding "Marlin has no SM100 SASS target, so on
Blackwell it JITs Ampere PTX." If that is right, **first-launch PTX JIT of the
Marlin kernels is an unlisted startup cost for Kimi only**, and it is *not*
covered by any of the caches in §1 or §2 — it is the CUDA driver's own JIT cache,
`CUDA_CACHE_PATH` (default `~/.nv/ComputeCache`, `CUDA_CACHE_MAXSIZE` default
256 MiB). **Persist `CUDA_CACHE_PATH` on weka and raise `CUDA_CACHE_MAXSIZE`** —
see §6.2.

#### 5.2.3 Kimi's warmups are mostly no-ops — and its ViT is not

**[D]** Two of the warmup steps whose names suggest Kimi relevance do nothing here:

- `kimi_k3_triton_warmup` returns immediately unless a `KimiK3DeltaAttention`
  layer is present (`warmup/kimi_k3_triton_warmup.py:22-40`). That is Kimi-**K3**'s
  KDA linear attention, a different model family (`vllm/models/kimi_k3/`).
  K2.6 is dense MLA. **No-op.**
- `sparse_mla_triton_warmup` only fires for the sparse backend names listed in
  `warmup/sparse_mla_triton_warmup.py:17-33`. Dense MLA matches none. **No-op.**
- Likewise `flashinfer_sparse_mla_decode_autotune_warmup` acts only on
  `FLASHINFER_MLA_SPARSE_SM120` / `FLASHINFER_MLA_SPARSE_DSV4`
  (`warmup/flashinfer_sparse_mla_warmup.py:35-41`). **No-op** — and note this is
  also a **correction to §2.2 for B300 generally**: that step is an SM120/DSv4
  path, not something our B300 boots pay.

What Kimi *does* pay is the generic `flashinfer_autotune()` step — a full
`_dummy_run` at `max_num_batched_tokens` through all 61 layers under
`AutoTuner(tune_mode=True)` — plus:

**[R] Issue #52965** (open): *"bare `@torch.compile` in `kimi_k25_vit` compiles
outside the compilation lifecycle and pins `TRITON_CACHE_DIR` process-wide."*
**[D]** I confirmed the decorator: `models/kimi_k25_vit.py:64` is a bare
`@torch.compile(dynamic=True, ...)`, not `@support_torch_compile`. Two
consequences: the ViT's Inductor output is **not** in the AOT artifact (so §1's
`VLLM_FORCE_AOT_LOAD=1` canary will not cover it), and it can seize
`TRITON_CACHE_DIR` for the process. **[I] This is the concrete reason §1.6's fix
(export `TRITON_CACHE_DIR` explicitly) is mandatory rather than nice-to-have for
Kimi.**

#### 5.2.4 `--trust-remote-code`: not for the config, possibly for the processor

**[D]** The repo ships 9 `.py` files (`configuration_kimi_k25.py`,
`modeling_kimi_k25.py`, `configuration_deepseek.py`, `modeling_deepseek.py`,
`tokenization_kimi.py`, `kimi_k25_processor.py`, `kimi_k25_vision_processing.py`,
`media_utils.py`, `tool_declaration_ts.py`) and an `auto_map` pointing at them.
vLLM bundles `KimiK25Config` (`transformers_utils/config.py:104`) and registers
the architecture in-tree (`registry.py:475`), so the **config** path does not need
remote code. **[D]** The tokenizer's tiktoken BPE vocab is bundled in the repo as
`tiktoken.model` (2.79 MB), so there is **no live network call** in the tokenizer
path once the snapshot is local. **[I]** The multimodal *processor* may still pull
`kimi_k25_processor.py` through `auto_map`; if you serve text-only, combine
`--limit-mm-per-prompt '{"image":0,"video":0}'` with dropping
`--trust-remote-code` and see whether it still boots. Moonshot's own
`deploy_guidance.md` passes `--trust-remote-code`, but their reference stack is
vLLM 0.19.1, well behind ours.

**[D] Moonshot's published command** (`docs/deploy_guidance.md`, 8xH200):
`vllm serve $MODEL -tp 8 --mm-encoder-tp-mode data --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2`.
It contains **no startup-time guidance whatsoever** — neither Moonshot's docs nor
the vLLM recipes page for K2.5 mention weight-load time, the Marlin repack, or
the INT4 path on NVIDIA hardware. That is a documentation gap, not a solved
problem.

---

### 5.3 deepseek-ai/DeepSeek-V3.2-Exp — DeepGEMM is not optional, and we did not turn it off

**What it is.** `DeepseekV32ForCausalLM` → `deepseek_v2.DeepseekV3ForCausalLM`
(`registry.py:95`). 61 layers, 256 routed experts top-8, MLA plus the DSA
lightning indexer (`index_topk: 2048`, `index_n_heads: 64`, `index_head_dim: 128`),
FP8 block-scale `[128,128]` with **`scale_fmt: ue8m0`**, `num_nextn_predict_layers: 1`.
**163 safetensors shards** — the most of the four, so the most exposure to the
per-file convoy of §4.7.

**[D]** No `auto_map`; the `.py` files in the repo live under `inference/` and are
DeepSeek's standalone reference implementation, not remote code vLLM loads.
`--trust-remote-code` is unnecessary. **[D]** `--tokenizer-mode deepseek_v32` is
selected automatically (`config/model.py:683`).

#### 5.3.1 `VLLM_USE_DEEP_GEMM=0` does not do what our recipe comment says

This is the most consequential finding in §5.

**[D]** The DSA indexer's CUDA custom op **hard-requires the DeepGEMM package at
construction time**:

```python
# vllm/model_executor/layers/sparse_attn_indexer.py:774-778
if current_platform.is_cuda() and not has_deep_gemm():
    raise RuntimeError(
        "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
        "the current vLLM environment.")
```

**[D] `has_deep_gemm()` is an import check** (`vllm/utils/import_utils.py`), not
the env var. The env var appears only in
`is_deep_gemm_supported() = envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and support_deep_gemm()`
(`utils/deep_gemm.py:105-110`), which is what gates the **warmup**:

```python
# vllm/model_executor/warmup/kernel_warmup.py
do_deep_gemm_warmup = (envs.VLLM_USE_DEEP_GEMM
                       and is_deep_gemm_supported()
                       and envs.VLLM_DEEP_GEMM_WARMUP != "skip")
```

**[D] So on DeepSeek-V3.2 and GLM-5.2, `VLLM_USE_DEEP_GEMM=0` does not disable
DeepGEMM. It disables the *warmup* of DeepGEMM.** The indexer still calls
`fp8_mqa_logits` (prefill) and `fp8_paged_mqa_logits` (decode) through
`vllm/v1/attention/backends/mla/indexer.py`, and DeepGEMM still JIT-compiles them
— on the **first real request** instead of during startup.

**[I] Practical consequence for us.** Our current recipe is, unintentionally, the
"fast start, slow first request" configuration for the two DSA models. That is
arguably the right choice for a throughput sweep and the wrong one for a latency
measurement, but either way the cost is real and currently invisible in our
time-to-ready number. Two coherent positions:

```bash
# A. Honest fast-start (what we have, but make it explicit and keep the cache):
VLLM_DEEP_GEMM_WARMUP=skip   DG_JIT_CACHE_DIR=/weka/.../dg-jit
#    ...then send one synthetic prefill + one decode before declaring "ready".
# B. Pay it once at boot, then never again (better for a served endpoint):
VLLM_USE_DEEP_GEMM=1  VLLM_DEEP_GEMM_WARMUP=relax  DG_JIT_CACHE_DIR=/weka/.../dg-jit
```
Note `VLLM_USE_DEEP_GEMM` and `VLLM_DEEP_GEMM_WARMUP` are both compile-hash
factors (§1.2), so pick one and freeze it.

#### 5.3.2 How bad is the DeepGEMM warmup, and is its cache portable?

**[R] vLLM issue #32116** (open): DeepSeek-V3.2, 4x8 H200, `-tp 2 -dp 16`:
`DeepGEMM warmup: 100%|████| 8181/8181 [19:02<00:00, 7.16it/s]` — **19 minutes**,
long enough that the API server's handshake with the engine core timed out and
the API server exited while the engine cores kept warming. The 0.28.0 source is
blunt about it: the `VLLM_DEEP_GEMM_WARMUP` docstring says this warmup "increases
the engine startup time by a couple of minutes."

**[D]** DeepGEMM ships **no precompiled cubins for the GEMM/MQA kernels** — its
README states all kernels are compiled at runtime through DeepJIT. There is no
`flashinfer-cubin` equivalent. The only levers are the warmup mode and the cache.

**[R] Two cache hazards worth testing before trusting a shared weka cache:**

1. DeepGEMM's C++ `init()` reads `DG_JIT_CACHE_DIR` **at import time**; if it is
   unset or the directory does not exist then, the persistent cache is disabled
   for the life of the process (in-memory only). vLLM sets it in
   `utils/deep_gemm.py:261-263` inside `_lazy_init()`, which runs *later* than
   some quantization code paths that `import deep_gemm`. vLLM PR **#39913**
   ("create DeepGEMM JIT cache directory before import", moving it to
   `env_override.py`) was **never merged**, and I confirmed `DG_JIT_CACHE_DIR`
   does not appear in `vllm/env_override.py` at v0.28.0. **Mitigation: export
   `DG_JIT_CACHE_DIR` yourself in the job script and `mkdir -p` it first.** It is
   not a `VLLM_*` var, so it is hash-neutral.
2. DeepGEMM upstream PRs **#388/#398** (cache key embedded the absolute
   `-I{include_path}`, so a renamed venv or a different container path re-JITs
   everything) and **#301/#302** (multi-process JIT cache race). Whether these are
   in the DeepGEMM version vLLM 0.28.0 pins is **unverified**. **[I] Until it is:
   keep the install prefix byte-identical across jobs** (see §6.1), and treat a
   suddenly-large `DG_JIT_CACHE_DIR` as the symptom of a path-dependent key.

#### 5.3.3 The rest of the DSA startup surface

- **[D] Triton metadata kernels.** The V3.2 backend name is `DEEPSEEK_V32_INDEXER`;
  `sparse_mla_triton_warmup` compiles only `_BUILD_PREFILL_CHUNK_METADATA_KERNEL`
  for it (`warmup/sparse_mla_triton_warmup.py:34,106-113`). The sparse-SWA and
  combine-topk kernels are DeepSeek-V4-only and are skipped. This step *is* behind
  `--kernel-config.enable_jit_warmup`.
- **[D] CuTe-DSL.** `fused_q_cutedsl.py` (fused Q + indexer-Q) and, when DCP>1,
  `dcp_indexer_cutedsl` via `_merge_dcp_topk_global` (`sparse_attn_indexer.py:102`).
  Our recipe runs `--decode-context-parallel-size 8`, so we are on the DCP CuTe-DSL
  path. These compile under `cutedsl_warmup()`; watch for the tqdm bar
  `Compiling CuTeDSL kernels` and the line
  `Warming up CuTeDSL compile_units=%d names=%s.`
- **[D] FlashMLA sparse kernels are precompiled**, shipped as a built extension by
  the `flash-mla` wheel, with SM90 and SM100 sparse decode/prefill targets listed
  in its support matrix. They are not a JIT cost.
- **[D] A second KV cache group.** `DeepseekV32IndexerCache` returns an
  `MLAAttentionSpec(num_kv_heads=1, head_size=132, dtype=torch.uint8)` alongside
  the main MLA latent cache, so the hybrid KV-cache coordinator has to size and
  allocate two heterogeneous groups during memory profiling. **[I]** More moving
  parts in exactly the phase that `VLLM_ENABLE_STARTUP_PLAN=1` lets you skip — a
  further reason to turn that on for this model.
- **[R] `--block-size 64` is effectively mandatory.** Issue **#48286**:
  *"DeepseekV32IndexerBackend requires `--block-size 64` — not documented or
  auto-detected."* FlashMLA's sparse kernels expect block size 64. Leave the
  default; do not sweep block size on this model.
- **[D] MTP disables CUDA graphs entirely.** `vllm/config/speculative.py:759-765`:
  ```python
  if self.method == "mtp":
      if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
          # FIXME(luccafong): cudagraph with v32 MTP is not supported,
          # remove this when the issue is fixed.
          self.enforce_eager = True
  ```
  **[I] So for DeepSeek-V3.2, `--speculative-config '{"method":"mtp",...}'` is
  simultaneously the largest startup *saving* available (all of §3 disappears) and
  a large steady-state loss.** It is also a silent one — nothing in the log says
  "your CUDA graphs were turned off by your speculative config." If you enable MTP
  on this model, expect a boot several minutes faster and decode meaningfully
  slower, and do not attribute either to anything else.

**[D] Published guidance.** DeepSeek's own repo points at the vLLM recipe, which
recommends `-dp 8 --enable-expert-parallel` over `-tp 8` ("the kernels are mainly
optimized for TP=1"), `--max-num-seqs 256` if you hit CUDA config errors, and
`VLLM_USE_DEEP_GEMM=0` as an H20 workaround — note that recommendation is about
the *MoE* path on H20 and, per §5.3.1, does not remove the indexer's DeepGEMM use.

---

### 5.4 zai-org/GLM-5.2-FP8 — DeepSeek's code path, plus B300-only kernels, minus a working compile story

**What it is.** `GlmMoeDsaForCausalLM` / `glm_moe_dsa`: 78 layers, 256 routed +
1 shared expert top-8, MLA (`q_lora_rank 2048`, `kv_lora_rank 512`,
`qk_nope 192`, `qk_rope 64`, `v_head_dim 256`), DSA with `index_topk: 2048`,
`index_n_heads: 32`, **`index_topk_freq: 4`** and `index_share_for_mtp_iteration: true`
(one real indexer computed per 4 layers; the other three share it),
`num_nextn_predict_layers: 1`, FP8 e4m3 `[128,128]`, vocab 154,880, 141 shards,
no remote code.

**[D] It is DeepSeek-V3.2's implementation.** In v0.28.0: `registry.py:118` maps
`GlmMoeDsaForCausalLM → ("deepseek_v2", "GlmMoeDsaForCausalLM")`, and
`deepseek_v2.py:1931` is literally `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`.
Every word of §5.3 — the DeepGEMM hard dependency, `VLLM_USE_DEEP_GEMM=0` being a
warmup switch only, the indexer Triton metadata kernel, the CuTe-DSL DCP merge,
the second KV group — applies verbatim, scaled by 78/61 layers. `deepseek_v2.py:127`
even special-cases `model_type == "glm_moe_dsa"` inline.

#### 5.4.1 The V1/V2 model-runner fork is the thing to watch

**[D]** There are two implementations. The V1 one (`model_executor/models/deepseek_v2.py`)
carries `@support_torch_compile` (`:1360`). The V2-model-runner one lives in the
separate `vllm/models/deepseek_v32/` tree, whose `__init__.py` says the same code
"serves any DSA checkpoint, including GLM-5.2 (`glm_moe_dsa`), which reuses this
architecture", and which on CUDA aliases `GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM`.
**[D] There is no `support_torch_compile` anywhere under `vllm/models/` in
v0.28.0** (`grep -rln support_torch_compile vllm/models/` → nothing).

**[D] Which one you get in 0.28.0:** `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
(`config/vllm.py:69-80`) contains `DeepseekV2ForCausalLM`, `DeepseekV4ForCausalLM`,
`GraniteMoeForCausalLM`, `Inkling*`, `KimiK3ForConditionalGeneration`,
`LongcatFlashNgramForCausalLM`, `Qwen2MoeForCausalLM` — **not**
`GlmMoeDsaForCausalLM` and **not** `DeepseekV32ForCausalLM`. And
`_is_default_v2_model_runner_model` (`config/vllm.py:690-716`) ends
`return is_default_v2_architecture or not model_config.is_moe`. Both models are
MoE and neither name is in the set, so **on stock v0.28.0 both route to the V1
runner and both compile.**

**[R] That changes on main.** Issue **#54197** (open, filed 2026-08-28 against
`main`): GlmMoeDsa default-routes to V2, `vllm/models/deepseek_v32/nvidia/model.py`
has no `@support_torch_compile`, and the engine prints
`torch.compile is turned on, but the model ... does not support it` and **silently
runs eager**. Force-enabling it then fails Dynamo `fullgraph` on
`is_fused_q_cutedsl_supported` (an `@lru_cache`d `has_device_capability`) and on a
`ContextVar.get()` in `v1/worker/workspace.py`. PR **#49790** ("Route DSA models
to the SM100 implementation") is the change that moves `GlmMoeDsaForCausalLM`
onto `vllm.models.deepseek_v32`.

**[I] Two things follow.** (1) Do not set `VLLM_USE_V2_MODEL_RUNNER=1` for GLM on
0.28.0 expecting a win — you trade a compiled model for an eager one and your
compile cache becomes dead weight. (2) **When we upgrade past 0.28.0, GLM's
startup will get faster and its decode slower, for reasons that appear in the log
only as one `warning_once`.** Grep for `does not support it` in §6.6's checklist.

#### 5.4.2 GLM-5.2 has kernels that exist for no other model, targeting exactly our GPU

**[D]** `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` — docstring:
*"GLM-5.2 decode GEMM selection for unquantized BF16 on SM103."* It hard-codes
CuTe-DSL `SkinnyGemmConfig`s for three projections — `GLM52_QKV_A_PROJECTION`
(n=2624, k=6144), `GLM52_Q_B_PROJECTION` (2048x2048) and `GLM52_EH_PROJECTION` —
selected per token count, with a `dsv3_fused_a` fallback for M in 3..16 and a
comment that "cuBLAS wins from M=4". It is wired in at `nvidia/model.py:399-400`
and `nvidia/mtp.py:244-245` via `enable_glm52_low_latency_gemm`. **[R]** vLLM
v0.28.0 release notes list PR **#49791** "CuTe DSL skinny GEMM extended to GLM-5.2".

**[D]** SM103 is B300. `is_device_capability_family(100)` buckets `10.x` together,
so B300 gets the SM100 path.

**[I] Two consequences.** (a) GLM pays a CuTe-DSL compile surface on B300 that it
does not pay on H200 — CuTe-DSL compiles through NVRTC/nvJitLink at process start
and has no vLLM-managed on-disk cache, so this cost recurs every boot. (b) These
kernels live in the `vllm/models/deepseek_v32/` (V2-runner) tree, so on stock
0.28.0 — where GLM routes to V1 (§5.4.1) — **we are probably not getting them at
all.** Confirm from the log: the `Warming up CuTeDSL compile_units=%d names=%s.`
line names the providers; if `skinny_gemm` is absent, the SM103 path is not active.

#### 5.4.3 GLM-specific landmines that overlap with startup

- **[R] #54300 (open):** *"[Regression 0.27→0.28+]: GlmMoeDsa (GLM-5.3) +
  decode-context-parallel: crashes on 0.28.0, silently returns random tokens on
  0.29.0."* **Our `README.md` recommends `--decode-context-parallel-size 8` for
  GLM and reports a measured DCP run as validation of the whole planner.** The
  issue is against GLM-5.3, but the arch is the same class. This needs an explicit
  output-correctness check on our exact build before the DCP result is trusted.
- **[R] #52150 (open):** GLM-5.2-FP8, first request after GPU idle emits garbage;
  piecewise CUDA-graph cold replay corrupts the request's own prefill; documented
  workaround `cudagraph_mode=FULL_DECODE_ONLY`. **[I] That is the same flag §3.3
  recommends for cutting capture time ~2x. For GLM the startup optimisation and the
  correctness workaround coincide — take it.**
- **[R] #49844 (open):** PP=2 + GlmMoeDsa: Inductor compile *combined with* CUDA
  graph capture produces garbage; either alone is clean.
- **[R] #53134 (open):** DCP unavailable for GlmMoeDsa on SM90 because the sparse
  MLA backend lacks decode-LSE support — not our arch, but it shows DCP
  availability is backend- and arch-conditional, not a property of the flag.
- **[R] B300/sm_103 toolchain risks, not GLM-specific but they surface during the
  profiling/autotune phase of startup:** #30245 (`PTXAS error: gpu-name sm_103a not
  defined` — Triton's ptxas too old), #30441 (Triton JIT autotune failing to build
  `cuda_utils.c` during `determine_available_memory` on B300 SXM6, CUDA 13.0),
  #30630 (`SymmMemCommunicator: Device capability 10.3 not supported`). **[I]
  Validate `ptxas --version` knows `sm_103a` in the image before blaming the model.**

**[D] Published guidance.** The vLLM recipe for GLM-5.2 gives
`--kv-cache-dtype fp8 -tp 8 --speculative-config.method mtp
--speculative-config.num_speculative_tokens 5 --tool-call-parser glm47
--reasoning-parser glm45 --enable-auto-tool-choice`, and offers an explicit
**"faster startup"** variant that is just `VLLM_DEEP_GEMM_WARMUP=skip` in front of
the same command, annotated *"skips DeepGEMM JIT warmup for a faster startup; the
first few requests compile kernels on demand instead."* Its troubleshooting
section notes FP8 performance *requires* DeepGEMM, installed from source via
`install_deepgemm.sh` — i.e. a build step that is not in the pip install and must
be in the image.

---

### 5.5 What this changes in our current launch recipe

| Current (`scripts/serving/README.md`) | Change | Why |
|---|---|---|
| `--trust-remote-code` on DeepSeek, GLM, Kimi | **Drop it** for DeepSeek and GLM; test dropping it for Kimi | §5.0 — vLLM bundles all four config classes; DeepSeek/GLM repos contain no `auto_map` at all |
| `--tokenizer-mode deepseek_v32` | Drop (redundant) | `config/model.py:683` auto-selects it |
| `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0  # JIT needs nvcc` | **The comment is wrong for DeepSeek and GLM.** Keep the flag if you want a fast boot, but say what it does: it moves DeepGEMM's JIT to the first request | §5.3.1 — `has_deep_gemm()` is an import check; the indexer raises without the package |
| (nothing) | Add `DG_JIT_CACHE_DIR=<weka>` and `mkdir -p` it **before** `vllm serve` | §5.3.2 — DeepGEMM disables its persistent cache if the dir is missing at import |
| (nothing) | Add `CUDA_CACHE_PATH=<weka>` + `CUDA_CACHE_MAXSIZE=4294967296` | §5.2.2 — driver-level PTX JIT cache, the only cache covering Marlin on a new arch |
| `--enable-prefix-caching` on Qwen | Verify correctness first | §5.1.6 — #55766 NaN after prefix-cache hit; #51198/#51250 0% hit rate |
| `--decode-context-parallel-size 8` on GLM | Verify output correctness on our build | §5.4.3 — #54300 |
| (nothing) | `--kernel-config.enable_flashinfer_autotune=false` for config-search runs | §2.2/§2.4 — hash-neutral, and the autotune is a full max-batch dummy run through 61–78 layers |
| Qwen `--language-model-only` | Keep, and **assert** the `init engine ... (compilation: ...)` line has no `encoder:` term | §5.1.4 |
| (nothing) | For Qwen config-search runs: `--additional-config '{"gdn_prefill_backend":"triton"}'` | §5.1.2 — the B300 default silently JITs FlashInfer GDN prefill |

---

## 6. A concrete warm-start procedure for Beaker/gantry

**Premise.** The container filesystem is destroyed between jobs. `/weka/oe-adapt-default`
and `/weka/oe-training-default` persist (`mason.py:355-362`). Several jobs may run
concurrently on different nodes against the same weka paths. Nothing below assumes
a privileged image or a patched vLLM.

Paths in this section are written against a single root you set once:

```bash
export VLLM_WARM=/weka/oe-adapt-default/allennlp/vllm-warm
```

### 6.0 The three classes of state

1. **Portable and shareable.** Derived from (model config, vLLM build, torch build,
   GPU arch) and written with atomic rename. Safe to share read-write between
   concurrent jobs. This is the torch.compile/AOT cache, the Inductor cache, the
   startup plan, the FlashInfer autotune cache.
2. **Portable but lock-serialised.** Correct to share, but concurrent writers
   *serialise on a file lock* or risk a half-written build tree. FlashInfer's JIT
   dir and DeepGEMM's JIT cache are here. Share them **read-only** at steady state;
   write to them only from the warm-up job.
3. **Not persistable at all.** Must be rebuilt every job: CUDA graphs, the NCCL/EP
   communicator setup, the OS page cache, Marlin-repacked weights, CuTe-DSL
   compilation, and any `@torch.compile` that runs outside vLLM's lifecycle
   (§5.2.3).

### 6.1 What goes on weka, and what must not be shared

| Path under `$VLLM_WARM` | Set via | Contents | Concurrency |
|---|---|---|---|
| `vllm-cache/` | `VLLM_CACHE_ROOT` | `torch_compile_cache/`, `startup_plan/`, `deep_gemm/`, `flashinfer_autotune_cache/` | **Shared RW.** vLLM writes every one of these with `os.replace` after a temp write (`decorators.py:702-706`, `startup_plan.py:186-189`, `flashinfer_autotune_cache.py:44-58`, `compiler_interface.py:210-248` patches `CompiledArtifact.save` for atomicity). POSIX rename on wekafs is atomic, so a concurrent reader sees old-or-new, never torn. |
| `triton/` | `TRITON_CACHE_DIR` | Triton JIT + autotune results (`TRITON_CACHE_AUTOTUNING=1` is forced, `env_override.py:113`) | **Shared RW.** Not set by vLLM on the AOT path (§1.6) — you must export it. |
| `inductor/` | *(do not set)* | — | **Leave it alone.** vLLM sets `TORCHINDUCTOR_CACHE_DIR` itself, per compile-hash, inside `VLLM_CACHE_ROOT` (`decorators.py:550-559`). Setting it yourself fights that. |
| `flashinfer-home/` | `FLASHINFER_WORKSPACE_BASE` | `.cache/flashinfer/{version}/{arch}/cached_ops`, `generated`, `cubins` | **Warm-up job writes; serving jobs read.** See §6.1.1. |
| `dg-jit/` | `DG_JIT_CACHE_DIR` | DeepGEMM DeepJIT kernels | **Warm-up job writes; serving jobs read.** See §6.1.2. |
| `nv-compute-cache/` | `CUDA_CACHE_PATH` (+ `CUDA_CACHE_MAXSIZE`) | driver-level PTX→SASS JIT cache | **Per-job copy.** See §6.1.3. |
| `hf/` | `HF_HOME`, `HF_HUB_CACHE` | model snapshots | Shared RO. `mason.py:319-323` already points these at `/weka/oe-adapt-default/allennlp/.cache/...`; keep those, and add `HF_HUB_OFFLINE=1` once the snapshot exists. |
| `logs/` | — | one `startup-<jobid>.jsonl` per job | append-only, per-job filename |

#### 6.1.1 FlashInfer's JIT dir is the one real corruption/serialisation hazard

**[D]** FlashInfer guards every build with `filelock.FileLock`, and the lock lives
*inside the cache tree*: `JitSpec.lock_path = get_tmpdir() / f"{name}.lock"` where
`get_tmpdir()` is `FLASHINFER_JIT_DIR / "tmp"` (`flashinfer/jit/core.py:387-388,
633-637`). The build itself takes a second, coarser lock,
`FileLock(tmpdir / "flashinfer_jit.lock")`, around the whole ninja invocation
(`core.py:662-665`). FlashInfer's own source carries the comment:

```python
def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.FLASHINFER_JIT_DIR / "tmp"
```

**[I] Two hazards follow, and they are different.**
- *Serialisation:* if eight jobs on eight nodes share `FLASHINFER_JIT_DIR` and all
  need to build the same module, they queue on one `flock` over wekafs. With a warm
  cache this never triggers (the fast path returns before the lock matters for
  loading), but on a cold cache your eight-way parallel sweep becomes serial.
- *Destruction:* `flashinfer.jit.core.clear_cache_dir()` does
  `shutil.rmtree(FLASHINFER_JIT_DIR)` unconditionally (`core.py:122-126`). Anything
  that calls it — a CLI command, a teardown path — wipes the shared tree out from
  under every concurrent job.

**Mitigation:** the warm-up job (§6.4) is the only writer. Serving jobs get the
same tree, and the ninja/lock path never activates because
`flashinfer-jit-cache`'s AOT modules are found first (`core.py:375-380,417-424`).
If you want belt-and-braces, `cp -a` the tree into the job's own scratch and point
`FLASHINFER_WORKSPACE_BASE` there — it is version+arch keyed, so a copy is valid.

#### 6.1.2 DeepGEMM: set the dir *before* anything imports it

**[D]** vLLM sets `DG_JIT_CACHE_DIR` only inside `_lazy_init()`
(`utils/deep_gemm.py:260-266`), which can run after a quantization path has already
`import deep_gemm`. DeepGEMM reads the variable at import and, if it is unset or
the directory does not exist, **silently drops to an in-memory-only cache for the
life of the process** — a warm weka cache then buys nothing and you will not be
told. The fix (PR #39913, moving it to `env_override.py`) was never merged; I
confirmed `DG_JIT_CACHE_DIR` is absent from `vllm/env_override.py` at v0.28.0.

```bash
export DG_JIT_CACHE_DIR="$VLLM_WARM/dg-jit"
mkdir -p "$DG_JIT_CACHE_DIR"     # must exist BEFORE the python process starts
```

**[R] Unverified risk, test before sharing RW:** DeepGEMM upstream PRs #301/#302
(multi-process JIT-cache race) and #388/#398 (cache key embedded the absolute
include path, so a different install prefix re-JITs everything). Whether they are
in the version vLLM 0.28.0 pins is unknown. **[I]** Therefore: warm-up job writes,
serving jobs read; and keep the install prefix byte-identical across jobs (§6.2).

#### 6.1.3 `CUDA_CACHE_PATH` — per-job copy, never shared

The driver's PTX→SASS cache is the only thing that covers kernels shipped as PTX
and JIT-compiled by the driver on first launch — which, if our earlier finding
that Marlin has no SM100 SASS target holds, is exactly Kimi's MoE kernels (§5.2.2).
**[I]** The driver's cache is an opaque, driver-version-keyed store with its own
locking that is not designed for a shared network filesystem. Treat it as
warm-once, copy-per-job:

```bash
export CUDA_CACHE_MAXSIZE=4294967296          # 4 GiB; default is 256 MiB
export CUDA_CACHE_PATH=/scratch/nv-cache      # node-local
mkdir -p "$CUDA_CACHE_PATH"
cp -a "$VLLM_WARM/nv-compute-cache/." "$CUDA_CACHE_PATH/" 2>/dev/null || true
```
and in the warm-up job only, copy it back at the end.

### 6.2 The environment block

Paste this **before** any `python`/`uvx` invocation. Order matters: everything that
another library reads at *import* time must be set first, and the directories must
already exist.

```bash
set -euo pipefail
export VLLM_WARM=/weka/oe-adapt-default/allennlp/vllm-warm

# --- 1. cache locations (all hash-neutral: none is a compile factor) ---------
export VLLM_CACHE_ROOT="$VLLM_WARM/vllm-cache"
export TRITON_CACHE_DIR="$VLLM_WARM/triton"
export FLASHINFER_WORKSPACE_BASE="$VLLM_WARM/flashinfer-home"
export DG_JIT_CACHE_DIR="$VLLM_WARM/dg-jit"
export CUDA_CACHE_PATH=/scratch/nv-cache
export CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" \
         "$DG_JIT_CACHE_DIR" "$CUDA_CACHE_PATH"
cp -a "$VLLM_WARM/nv-compute-cache/." "$CUDA_CACHE_PATH/" 2>/dev/null || true

# --- 2. things that must be identical across every job in a cache lineage ----
#     (each of these IS a torch.compile hash factor; changing one = cold compile)
export VLLM_ENABLE_STARTUP_PLAN=1        # ignored by the hash, but set it once anyway
export VLLM_ENGINE_READY_TIMEOUT_S=3600  # IS hashed -- pick one value forever
export VLLM_USE_DEEP_GEMM=1              # IS hashed -- see 5.3.1; pick a side
export VLLM_MOE_USE_DEEP_GEMM=1
export VLLM_DEEP_GEMM_WARMUP=relax       # IS hashed
export VLLM_USE_FLASHINFER_MOE_INT4=0    # IS hashed; flip only for a Kimi A/B lineage

# --- 3. correctness / non-hashed hygiene -------------------------------------
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$(( $(nproc) / 8 ))   # TP=8 -> one engine core per GPU; see 4.8
export TOKENIZERS_PARALLELISM=false

# --- 4. a stable install prefix (DeepGEMM & the legacy compile path key on paths)
export UV_CACHE_DIR="$VLLM_WARM/uv"
export VIRTUAL_ENV="$VLLM_WARM/venv-0.28.0"   # created once by the warm-up job
export PATH="$VIRTUAL_ENV/bin:$PATH"
```

**[D] Why the prefix must be stable.** The AOT compile key does not contain
absolute paths (`decorators.py:255-262`) — but the *legacy* compile path's
`code_hash` does (`backends.py:1040-1054`), and DeepGEMM's key did until
upstream #398. A fixed `VIRTUAL_ENV` on weka costs nothing and removes a whole
class of silent cache misses. It also removes the `uvx` resolve step from every
boot.

### 6.3 What must be rebuilt every job, and why

| Rebuilt each job | Cost | Why it cannot be cached |
|---|---|---|
| **CUDA graph capture** | **[R]** measured **80 s and 4.88 GiB/GPU** on B300 SXM6 to capture up to 1024 (vLLM PR #49390, the PR that made 1024 the Blackwell default) — for one descriptor set on a DeepSeek-class model | §3.2 — graphs hold device pointers into the live memory pool; CUDA has no portable serialisation |
| **NCCL / EP communicator setup** | seconds to tens of seconds; grows with `--enable-expert-parallel` (DeepEP buffer allocation) | per-process, per-topology |
| **OS page cache** | this is the §4 weight-load phase | node-local, dropped at container exit |
| **`process_weights_after_loading`** (Marlin repack, FP8 requant) | dominant for Kimi (§5.2.2); small for the FP8 three | no on-disk cache exists in 0.28.0; only `--load-format sharded_state` sidesteps it, at the cost of TP-locking |
| **CuTe-DSL compilation** | GLM's SM103 skinny GEMMs, DeepSeek's fused-Q and DCP merge (§5.3.3/§5.4.2) | compiled via NVRTC/nvJitLink at process start; no vLLM-managed on-disk cache |
| **Memory profiling + CUDA-graph memory estimation** | one full `_dummy_run` | **cacheable** — this is exactly what `VLLM_ENABLE_STARTUP_PLAN=1` removes (§3.4) |
| **FlashInfer autotune** | one `_dummy_run` at `max_num_batched_tokens` | **cacheable**, with the TP>1 deadlock caveat of §2.4 |

### 6.4 Order of operations at job start

The ordering is not cosmetic — steps 1-3 must precede any Python import, and step 5
must precede step 6 or the loader outruns the prefetcher (§4.3).

```
1.  Export the §6.2 block; mkdir every cache dir.        # DeepGEMM reads DG_JIT_CACHE_DIR at import
2.  cp -a the driver PTX cache from weka to node-local.  # §6.1.3
3.  Sanity-check the toolchain, fail fast:
       ptxas --version | grep -q . && ptxas --list-gpu-arch 2>/dev/null | grep -q sm_103
       python -c "import flashinfer, deep_gemm"          # both must import
       flashinfer show-config                            # must list Sm103a cubins
    # §5.4.3: vLLM issues #30245/#30441/#30630 are all "ptxas/Triton does not know
    # sm_103", and they surface 20 minutes in, during memory profiling.
4.  Pre-warm the page cache SYNCHRONOUSLY and time it separately:
       find "$MODEL_DIR" -name '*.safetensors' -print0 \
         | xargs -0 -P 32 -I{} dd if={} of=/dev/null bs=64M status=none
    # This is §4.3's hardening. It also gives you a clean per-node storage
    # benchmark: if this phase is slow, fail the job before touching a GPU.
5.  Launch vllm serve with the per-model flags from §5.5 and
       --safetensors-load-strategy=prefetch
       --safetensors-prefetch-num-threads=32
       --safetensors-prefetch-block-size=67108864
6.  Poll /health. On first 200, send ONE synthetic request with a long prompt and
    a short generation, then one with a short prompt.  # §5.1.2 / §5.3.1: the
    # prefill-path GDN kernel and the DeepGEMM indexer kernels JIT on first use,
    # AFTER "ready". Declare readiness only after this completes.
7.  Emit the §6.6 grep summary to $VLLM_WARM/logs/startup-$BEAKER_JOB_ID.jsonl.
```

### 6.5 The warm-up job

Run this **once per (vLLM build, model, TP, DCP, max-model-len,
max-num-batched-tokens, cudagraph config, VLLM_\* environment)** tuple. It is the
only job that writes to the class-2 caches.

```bash
#!/usr/bin/env bash
# warm.sh -- populate $VLLM_WARM for one configuration. One GPU node, ~1 hour.
set -euo pipefail
source ./env_block.sh                 # exactly §6.2, unchanged

# (0) build the venv on weka, once, at a fixed prefix
if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv --python 3.12 "$VIRTUAL_ENV"
  uv pip install vllm==0.28.0
  CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
  uv pip install flashinfer-cubin==0.6.16.post3     --index-url https://flashinfer.ai/whl/
  uv pip install flashinfer-jit-cache==0.6.16.post3 --index-url https://flashinfer.ai/whl/$CU
  # DeepGEMM is a from-source build, not a pip package -- the vLLM GLM-5.2 recipe
  # points at install_deepgemm.sh. Do it here, not in the serving job.
fi

# (1) compile-cache warm build: weights do not affect compilation (see 1.4)
vllm serve "$MODEL" --load-format dummy \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-dtype fp8 --max-model-len 131072 \
  -cc.cudagraph_mode=FULL_DECODE_ONLY -cc.max_cudagraph_capture_size=256 \
  ... &                                # identical to the serving command except --load-format
curl --retry 200 --retry-delay 5 --retry-all-errors -sf localhost:8000/health
kill %1

# (2) real-weights pass: populates the startup plan, the FlashInfer autotune
#     cache, the DeepGEMM JIT cache and the Marlin/driver PTX cache, none of
#     which a dummy-weight boot can produce correctly.
vllm serve "$MODEL" <exact serving flags> &
curl --retry 400 --retry-delay 5 --retry-all-errors -sf localhost:8000/health
curl -s localhost:8000/v1/completions -d '{"model":"'"$MODEL"'","prompt":"'"$(head -c 40000 /usr/share/dict/words | tr '\n' ' ')"'","max_tokens":8}'
kill %2

# (3) publish the driver cache back to weka
mkdir -p "$VLLM_WARM/nv-compute-cache"
cp -a "$CUDA_CACHE_PATH/." "$VLLM_WARM/nv-compute-cache/"

# (4) prove it: this boot MUST NOT recompile
VLLM_FORCE_AOT_LOAD=1 vllm serve "$MODEL" <exact serving flags>   # fails loudly on a miss
```

**[I] Why two passes.** Pass (1) is cheap (no 600 GB read) and produces the
torch.compile/AOT artifacts, which are weight-independent. Pass (2) cannot be
skipped because the startup plan records a *profiled* memory number, the
FlashInfer autotune cache records tactics chosen against real shapes, and
DeepGEMM/Marlin/driver-PTX caches only fill on real kernel launches. Pass (4) is
the regression gate: `VLLM_FORCE_AOT_LOAD=1` turns a silent recompile into a
startup failure (`decorators.py:326-327`).

**[D] One caveat on pass (1):** `--load-format dummy` is a `LoadConfig` change and
`LoadConfig.compute_hash` returns a constant (`config/load.py`), so the key is
right. What is *not* proven is that no quantization path specialises the graph on
values read from the checkpoint; pass (4) is what proves it, per config.

### 6.6 Detecting a cold or invalidated cache from the log

Every line below is a literal format string from the v0.28.0 tree. Grep for them
and emit a one-line verdict; a regression then shows up in the job log within
seconds instead of as an unexplained 40 minutes.

| Grep for | Source | Verdict |
|---|---|---|
| `Directly load AOT compilation from path` | `compilation/decorators.py:312` | **HIT** — AOT artifact reused, Dynamo skipped |
| `Dynamo bytecode transform time: ` | `compilation/backends.py:1156` | **MISS** — you are tracing from scratch |
| `Compiling a graph for compile range` | `backends.py:394` | **MISS** — Inductor is running |
| `Directly load the compiled graph(s) for compile range` | `backends.py:293` | partial hit (legacy path) |
| `Using cache directory: %s for vLLM's torch.compile` | `backends.py:1095` | prints the key dir — log it, diff it between jobs to see *what* changed |
| `Applying persisted startup plan (fingerprint %s)` | `v1/worker/startup_plan.py:154` | **HIT** — profiling skipped |
| `Startup plan not applied: current free memory` | `startup_plan.py:124` | plan rejected by the free-memory gate — a co-tenant or a leak |
| `Saved startup plan to %s` | `startup_plan.py:189` | first boot of this fingerprint |
| `Using FlashInfer autotune cache file: %s` | `warmup/kernel_warmup.py` | autotune cache located. **If the log stops here, you have hit the §2.4 TP>1 deadlock** |
| `Warming up CuTeDSL compile_units=%d names=%s.` | `warmup/cutedsl_warmup.py:108` | lists which CuTe-DSL providers are active — for GLM this is how you tell whether the SM103 skinny GEMMs (§5.4.2) are in play |
| `Compiling CuTeDSL kernels` (tqdm) | `cutedsl_warmup.py:85` | CuTe-DSL compile in progress |
| `DeepGEMM warmup` (tqdm) + `Deep GEMM warmup` span | `warmup/deep_gemm_warmup.py` | the §5.3.2 19-minute risk |
| `Warming up Qwen Triton kernels for model_type=%s.` | `warmup/qwen_triton_warmup.py` | Qwen GDN warmup ran |
| `Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.` | same | it did **not** — check the model type |
| `Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).` | `qwen_gdn_linear_attn.py:159` | `FlashInfer` here on a cold FlashInfer cache = a multi-minute silent JIT (§5.1.2) |
| `` `torch.compile` is turned on, but the model `` ... `does not support it` | `config/vllm.py:2635-2640` | **your compile cache is dead weight** — the §5.4.1 V2-runner trap |
| `Prefetching checkpoint files into page cache started (in background, num_threads=%d, block_size=%d bytes)` | `weight_utils.py:821-825` | confirms the flags took effect |
| `Prefetching checkpoint files into page cache finished in %.2fs` | `weight_utils.py:816-818` | time this against step 4 of §6.4 |
| `Auto-prefetch is disabled because the filesystem` | `weight_utils.py` | you forgot `--safetensors-load-strategy` (§4.1) |
| `EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts` | `default_loader.py:407` | EP filter engaged |
| `Graph capturing finished in %.0f secs, took %.2f GiB` | `gpu_model_runner.py:7049` | the irreducible §3 floor |
| `init engine (profile, create kv cache, warmup model) took %.2f s (compilation: %.2f s)` | `v1/engine/core.py:347-352` | the headline number. The three-term variant with `encoder:` means the vision tower compiled (§5.1.4) |

**A stronger instrument than grep.** vLLM 0.28.0 is OpenTelemetry-instrumented with
named spans covering exactly the phases in question — `Overall Loading`
(`v1/engine/core_client.py:115`), `Prepare model` (`core.py:252`), `Worker init`,
`Init device`, `Load weights` (`default_loader.py:414`), `Initialize model`
(`model_loader/utils.py:36`), `Loading (GPU)` (`gpu_model_runner.py:5413`),
`Compile graph` (`backends.py:263`), `Inductor compilation` (`backends.py:726`),
`DeepGemm warmup`, `CuTeDSL warmup`, `Allocate KV cache`, `Warmup (GPU)`,
`Capture model` (`gpu_model_runner.py:6948`). Install
`opentelemetry-sdk opentelemetry-exporter-otlp`, run a collector as a sidecar and
pass `--otlp-traces-endpoint`; workers self-register
(`multiproc_executor.py:902`, `core.py:1289`). **[I]** One afternoon of setup
turns "startup took 40 minutes" into a waterfall that says which phase, per rank.

**And a purpose-built benchmark.** `vllm bench startup` exists
(`vllm/benchmarks/startup.py`, CLI `entrypoints/cli/benchmark/startup.py`). Its
docstring: *"measures total startup time ... for both cold and warm scenarios —
Cold startup: Fresh start with no caches (temporary cache directories); Warm
startup: Using cached compilation and model info."* Options
`--num-iters-cold` (3), `--num-iters-warmup` (1), `--num-iters-warm` (3),
`--output-json`. It reports `cold_startup` / `warm_startup` /
`{cold,warm}_compilation` / `{cold,warm}_encoder_compilation` with percentiles.
**Use this to produce the numbers in §6.7 for real, rather than trusting the
estimates below.**

### 6.7 Expected best-case time-to-ready after warming

**[I] Everything in this table is an estimate, not a measurement.** It is built
from: our own measured ~5 min prefetch + ~7 min load for a 400-750 GB checkpoint;
the one published B300 capture measurement (80 s / 4.88 GiB for `max_capture=1024`,
one descriptor set, PR #49390); and the per-model structure established in §5. The
band is wide on purpose. **Replace it with `vllm bench startup --output-json`
output as soon as the warm caches exist.**

Assumed warm configuration: all §6.2 caches populated, `VLLM_ENABLE_STARTUP_PLAN=1`
applied, `flashinfer-cubin` + `flashinfer-jit-cache` installed, page cache
pre-warmed synchronously (§6.4 step 4), `-cc.cudagraph_mode=FULL_DECODE_ONLY`,
`-cc.max_cudagraph_capture_size=256`.

| Phase | Qwen3.5-397B | Kimi-K2.6 | DeepSeek-V3.2 | GLM-5.2 |
|---|---|---|---|---|
| process + import + config | 20–40 s | 30–60 s (9 remote-code files if `--trust-remote-code` kept) | 20–40 s | 20–40 s |
| weight read (page cache warm) | 60–120 s (406 GB, 94 shards) | 90–180 s (532–595 GB, 64 shards, 208 k tensors) | 100–200 s (689 GB, 163 shards) | 100–200 s (755 GB, 141 shards) |
| `process_weights_after_loading` | 10–30 s | **300–900 s** (§5.2.2, ~92 k launch-bound iterations/rank) | 20–60 s | 30–80 s |
| compile (AOT cache hit) | 10–30 s | n/a (LM not decorated) | 10–30 s | 10–40 s |
| kernel JIT (all caches warm) | 20–60 s | 30–90 s (FlashInfer autotune dummy run) | 30–90 s | 40–120 s (+ CuTe-DSL, uncached) |
| memory profiling | ~0 (startup plan) | ~0 | ~0 | ~0 |
| CUDA graph capture | 40–90 s | 40–90 s | 40–90 s | 50–110 s (78 layers) |
| **Estimated time-to-ready** | **3–6 min** | **9–22 min** | **4–8 min** | **4–10 min** |
| **vs. our current 40–55 min** | ~10x | ~3x | ~7x | ~6x |

**[I] Reading this table.**
- **Kimi is a different problem from the other three.** Its floor is set by a
  Python loop, not by a cache you can warm. Nothing in §1, §2 or §6 moves it. The
  only levers are `VLLM_USE_FLASHINFER_MOE_INT4=1` (§5.2.1, unproven), a
  `sharded_state` dump for a frozen config (§4.5), or the NVFP4 checkpoint our
  `README.md` already recommends for B300 — `nvidia/Kimi-K2.6-NVFP4` sidesteps the
  Marlin path entirely. **If Kimi startup matters, changing the checkpoint is the
  highest-leverage move available.**
- **The biggest single win for the other three is §2.1**, not anything in this
  section: installing `flashinfer-cubin` and `flashinfer-jit-cache` converts the
  dominant 40–55 min term into a file read. Everything in §6 is about making sure
  that file read keeps happening on the next job.
- **The floor is capture + load.** Once the caches hit, the irreducible cost is
  the weight read and the CUDA graph capture, and the only remaining lever on
  either is a smaller capture list or a faster loader (`instanttensor`, §4.5).

### 6.8 The one-line regression gate

Put this at the end of every serving job. If it prints anything, a cache broke.

```bash
grep -E 'Dynamo bytecode transform time|Compiling a graph for compile range|does not support it|Auto-prefetch is disabled|Startup plan not applied|Skipping Qwen GDN Triton warmup|Failed to connect to NVIDIA artifactory' \
     "$LOGFILE" || echo "OK: all caches hit"
```

The last pattern is the sleeper: **[D]** `has_nvidia_artifactory()`
(`vllm/utils/flashinfer.py:367-391`) does a live `requests.get` with a 5 s timeout
against `https://edge.urm.nvidia.com/artifactory/...` and, on failure, logs
`Failed to connect to NVIDIA artifactory: %s` and **silently drops the trtllm-gen
attention path**. It short-circuits to `True` only when the `flashinfer_cubin`
package is installed (`:373-375`). A Beaker node with restricted egress and no
`flashinfer-cubin` therefore degrades to a different — slower — attention backend
without failing, which is precisely the kind of regression that looks like noise
in a throughput sweep. Installing the cubin wheel (§2.1) fixes the startup cost
*and* removes the network dependency *and* removes this failure mode.
