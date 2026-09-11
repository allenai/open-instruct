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
