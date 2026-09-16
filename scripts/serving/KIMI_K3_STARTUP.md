# Minimising time-to-ready for `moonshotai/Kimi-K3` on 8x B300 (vLLM 0.28.0)

**Scope:** process launch -> `vllm ready`. Single node, 8x NVIDIA B300 SXM6 (sm_103,
275,040 MiB/GPU), weights on WEKAFS at `/weka/oe-adapt-default/shared/hf_cache`.

**Evidence tags used throughout:**

- **[D]** documented — read directly in the vLLM `v0.28.0` source tree, vLLM docs,
  the official vLLM recipe for this model, or the HuggingFace model config.
  File paths/line numbers refer to `github.com/vllm-project/vllm` tag `v0.28.0`.
- **[R]** reported by a third party (GitHub issue/PR, vendor blog, project README).
- **[I]** my inference. Treat as a hypothesis to measure, not as fact.

Everything below was read this session. Where a source was unreachable it says so.

---

## 0. The three facts that shape every recommendation

1. **[D] vLLM 0.28.0 has first-class Kimi-K3 support.** There is a whole
   `vllm/models/kimi_k3/` package (NVIDIA + AMD variants, KDA kernels, CuTe-DSL
   latent-MoE tail, MLA), a bundled config class
   (`vllm/transformers_utils/config.py:109` -> `kimi_k3="KimiK3Config"`), a
   dedicated tokenizer mode, reasoning parser, tool parser and Rust renderer.
   You are not on a fork or a nightly-only path.

2. **[D] The checkpoint is ~500k tensors, not ~500 tensors.**
   `config.json` (fetched from `huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`)
   gives `num_hidden_layers: 93`, `first_k_dense_replace: 1`, `num_experts: 896`,
   `quantization_config.format: "mxfp4-pack-quantized"`, `quant_method: compressed-tensors`.
   `model.safetensors.index.json` is **59,764,096 bytes** and
   `metadata.total_size` is **1,560,860,324,864 B**, with per-expert names of the form
   `language_model.model.layers.12.block_sparse_moe.experts.895.w3.weight_packed`
   and a sibling `...w3.weight_scale`.
   **[I]** 92 MoE layers x 896 experts x 3 projections x 2 tensors = **494,592
   expert tensors**, plus dense/attention/vision. The ~120 B/entry implied by the
   59.8 MB index corroborates ~500k entries.

3. **Your measurement is consistent with a per-tensor cost, not a bandwidth cost.**
   You measured 167 GB in 997–1398 s with load time *anti*-correlated with raw weka
   read bandwidth. **[I]** At the fastest node's measured 5,938 MB/s, the pure-I/O
   floor for K3 is `1561 GB / 5.938 GB/s ~= 263 s`. Anything above that is
   per-tensor overhead: page-fault + `get_tensor()` + narrow + H2D copy, repeated
   ~500k times per rank (under pure TP every rank touches every expert tensor and
   slices it). This is the same failure mode named in vLLM issue #52330, whose
   author notes the slowdown "scales with tensor count, not bytes" and is worst for
   packed INT4/NVFP4/MXFP4 checkpoints **[R]**.

Consequently the startup levers that matter most for K3 are (a) a loader that
amortises per-tensor work, (b) not re-doing JIT/autotune/profiling work that a
previous boot already did, and (c) *not* CUDA-graph trimming, which is a
throughput trade and a small share of the budget at this scale.

---

## 1. Recommended configuration

### 1.1 Container

**[D] Switch from `uvx` to the official image `vllm/vllm-openai:v0.28.0-x86_64`.**

From `docker/versions.json` and `docker/Dockerfile` at `v0.28.0`:
`CUDA_VERSION=13.0.3`, `UBUNTU_VERSION=24.04`, `PYTHON_VERSION=3.12`,
`NCCL_VERSION=2.30.7`, torch `2.13.0`.
Docker Hub confirms the tag exists (pushed 2026-08-26, 8.6 GB x86_64; the
suffix-less `v0.28.0` is the multi-arch manifest, and `-cu129` variants are the
CUDA 12.9 build you do *not* want).

What the image gives you that a `uvx vllm==0.28.0` install does not **[D]**:

| Component | Where it comes from | Why it matters at startup |
|---|---|---|
| `flashinfer-jit-cache==0.6.16.post3` | `Dockerfile:760-766`, installed from `https://flashinfer.ai/whl/cu130` | pre-built JIT modules; otherwise `nvcc` compiles at boot |
| `flashinfer-cubin==0.6.16.post3` | `requirements/cuda.txt` (`--extra-index-url https://flashinfer.ai/whl/`) | pre-compiled cubins; **excluded from the published wheel's `install_requires`**, per the comment in `cuda.txt` |
| `fastsafetensors >= 0.3.3` | `requirements/cuda.txt` | the loader the recipe wants (§1.3) |
| `tokenspeed-mla==0.1.8` | `requirements/cuda.txt` | required for `--attention-backend TOKENSPEED_MLA` |
| `nvidia-cutlass-dsl[cu13]==4.6.2`, `quack-kernels==0.6.4`, `tilelang==0.1.12`, `humming-kernels[cu13]==0.1.12`, `apache-tvm-ffi==0.1.11` | `requirements/cuda.txt` | K3's CuTe-DSL `gemm_rs` / `latent_moe_tail` path |
| `runai-model-streamer[s3,gcs,azure]>=0.15.7` | `Dockerfile:~830` | alternative loader, free to A/B |
| gdrcopy | `Dockerfile:775-790` | NCCL/NVLink init |

You already install `flashinfer-cubin` + `flashinfer-jit-cache` by hand, so the
image is not a step change *for FlashInfer specifically*; it is a step change for
everything else in that table, and it removes the in-job CUDA-toolkit install.

**sm_103 coverage is fine [D].** `docker/versions.json` sets
`TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0"` with no `10.3`, which
looks alarming, but `CMakeLists.txt:118-129` sets
`CUDA_SUPPORTED_ARCHS "...;10.0;10.1;10.3;12.0;12.1"` and, from CUDA 12.9 onward,
every Blackwell kernel target is written family-specific — e.g.
`CMakeLists.txt:465` `"9.0a;10.0f;10.1f;10.3f;10.7f;11.0f;12.0f;12.1f"`,
`:1031` FP4 `"10.0f;10.7f;11.0f"`, `:1097` MLA `"10.0f;10.7f;11.0f"` — and
`cmake/utils.cmake:416-445` resolves an `f` suffix against any target in the same
major family. A `10.0f` cubin runs on sm_103. The recipe independently marks
`b300: verified`.

**Caveat [D]:** the image does **not** contain `instanttensor` (it is not in
`requirements/cuda.txt`; `docs/models/extensions/instanttensor.md` says
`pip install instanttensor`). If you keep that loader you must add it to the image.

**Not recommended:** the `vllm/vllm-openai:kimi-k3` tag the recipe names. It was
pushed **2026-07-27**, a month before `v0.28.0` (2026-08-26), and is the
pre-release K3 preview (13.3 GB x86_64). K3 is fully merged now; the release tag is
newer and reproducible.

### 1.2 Environment

```bash
# ---- cache roots: everything below lands on weka and survives the job ----
export VLLM_CACHE_ROOT=/weka/oe-adapt-default/shared/vllm-cache
export TRITON_CACHE_DIR=/weka/oe-adapt-default/shared/vllm-cache/triton
export FLASHINFER_WORKSPACE_BASE=/weka/oe-adapt-default/shared/flashinfer-home
# DG_JIT_CACHE_DIR defaults to $VLLM_CACHE_ROOT/deep_gemm - no need to set it.

# ---- skip repeated profiling work on every boot after the first ----
export VLLM_ENABLE_STARTUP_PLAN=1

# ---- do not let the 600 s engine-ready deadline kill a 20-minute load ----
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- thread count for weight deserialisation. 244 CPUs / 8 local workers = 30.
#      Compute it rather than hardcoding: your 4-GPU job saw 56 CPUs, so the
#      cgroup quota an 8-GPU job gets is not necessarily 244.
export OMP_NUM_THREADS=$(( $(python -c 'import os;print(len(os.sched_getaffinity(0)))') / 8 ))

# ---- recipe-mandated for Blackwell ----
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_V2_MODEL_RUNNER=1

export HF_HOME=/weka/oe-adapt-default/shared/hf_cache
export HF_HUB_OFFLINE=1
```

### 1.3 Serve command

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --dcp-comm-backend a2a \
  --attention-backend TOKENSPEED_MLA \
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TRTLLM_RAGGED"}' \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --prefix-match-unit 128 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser kimi_k3 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --no-enable-flashinfer-autotune
```

**Deltas from your current command, and why:**

| Change | Why | Throughput risk |
|---|---|---|
| `--load-format instanttensor` -> `fastsafetensors` | the official recipe's Blackwell profile for this exact model **[D]**; the recipe comment reads "Canonical Inferact serve uses fastsafetensors for much faster weight load" | none (loader is compile-hash-neutral, §3) |
| **+** `--no-enable-flashinfer-autotune` | recipe Blackwell profile **[D]**; removes a full `_dummy_run(num_tokens=max_num_batched_tokens)` autotune sweep at boot (`warmup/kernel_warmup.py:flashinfer_autotune`) | **yes, flagged** — see §4 |
| **+** `--dcp-comm-backend a2a`, `--attention-backend TOKENSPEED_MLA`, `--attention-config {...}` | you already pass `--decode-context-parallel-size 8` and `--kv-cache-dtype fp8`; the recipe emits all five together and its notes say FP8 KV **requires** the `--attention-config` pair **[D]** | improves decode, does not slow startup |
| **+** `--prefix-match-unit 128` | recipe Blackwell profile **[D]**. K3 is hybrid (MLA + KDA), so vLLM inflates the attention block size to the Mamba page size and the default prefix-hit boundary becomes very coarse; 128 divides the inflated size | improves prefix-cache hit rate |
| **+** `--gpu-memory-utilization 0.95` | recipe `base_args` **[D]** (vLLM default is 0.9) | more KV cache |
| **-** `--tokenizer-mode kimi_k3` | redundant: `config/model.py:680-681` sets it automatically when `arch == "KimiK3ForConditionalGeneration"` **[D]** | none |
| keep `--trust-remote-code` | recipe `base_args` **[D]**. vLLM bundles `KimiK3Config` so the *config* does not need it, but the HF repo ships `auto_map` and the processor path is not something I verified offline — keep it until you have a clean boot to test against | none |

### 1.4 First-boot vs warm-boot

The startup plan and the compile/JIT caches only pay off on the **second** boot with
an identical config. Budget one cold boot per `(model, TP, DCP, kv-cache-dtype,
max-model-len, max-num-batched-tokens, cudagraph config, VLLM_* env)` tuple and then
freeze that tuple (§3).

---

## 2. Evidence table

| # | Recommendation | Source | Reported / observed effect | Confidence |
|---|---|---|---|---|
| 1 | `--load-format fastsafetensors` for K3 on Blackwell | `recipes/models/moonshotai/Kimi-K3.yaml`, `hardware_overrides.blackwell.extra_args` — https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml ; and https://vllm-project.github.io/2026/07/27/k3.html which prints the serve command with `--load-format fastsafetensors` | "much faster weight load" (no number published) | **[D]** that it is the blessed flag; **[I]** that it beats instanttensor on weka |
| 2 | `--no-enable-flashinfer-autotune` | same recipe, `hardware_overrides.blackwell`; implementation `vllm/model_executor/warmup/kernel_warmup.py` | removes `flashinfer_autotune()` -> one `_dummy_run` at `max_num_batched_tokens` with per-tactic timing + a world all-reduce per tactic | **[D]** |
| 3 | `VLLM_ENGINE_READY_TIMEOUT_S=3600` | recipe `hardware_overrides.blackwell.extra_env`; default is 600 s at `vllm/envs.py:27,800` | prevents `TimeoutError` killing a long load | **[D]** |
| 4 | `VLLM_ALLREDUCE_USE_FLASHINFER=1`, `VLLM_USE_V2_MODEL_RUNNER=1` | recipe `hardware_overrides.blackwell.extra_env`; vars exist at `envs.py:261,292` | steady-state, not startup | **[D]** |
| 5 | `--prefix-match-unit 128` | recipe + its `guide` note ("Prefix-match unit"); flag registered at `arg_utils.py:1243` | finer prefix-cache granularity on the hybrid KV manager | **[D]** |
| 6 | DCP flag set (`a2a` + `TOKENSPEED_MLA` + `--attention-config`) | recipe `features.decode_context_parallelism` | pairs TOKENSPEED_MLA decode with TRTLLM_RAGGED MLA prefill under FP8 KV | **[D]** |
| 7 | `VLLM_ENABLE_STARTUP_PLAN=1` | `vllm/v1/worker/startup_plan.py` (whole file read); env at `envs.py:263,1883` | persists the profiled `kv_cache_memory_bytes` + free-memory baseline under `$VLLM_CACHE_ROOT/startup_plan/`; later boots "skip the memory-profiling measurement and the CUDA-graph memory estimation pass". Fingerprinted on `VllmConfig.compute_hash()` + device name/memory/capability + torch/CUDA + rank + world size; gated on `current_free_memory >= baseline`; a stale plan is ignored | **[D]** |
| 8 | Startup plan is compile-hash-neutral | `envs.py:2229-2231` ignore list contains `VLLM_CACHE_ROOT` and `VLLM_ENABLE_STARTUP_PLAN` | you can turn it on without invalidating the torch.compile cache | **[D]** |
| 9 | `--kv-cache-memory-bytes` (do **not** hand-set; use #7 instead) | flag at `arg_utils.py:1211`; docs `docs/configuration/optimization.md:24` | "skips the memory-profiling measurement and the CUDA-graph memory estimation pass"; docs warn a low value "caps batch concurrency (and therefore throughput)" and a high one "fails at allocation time" | **[D]** |
| 10 | `VLLM_CACHE_ROOT` on weka | `docs/configuration/optimization.md:23`; `vllm/utils/deep_gemm.py` defaults `DG_JIT_CACHE_DIR` to `$VLLM_CACHE_ROOT/deep_gemm` | compile cache "can be copied between machines or baked into a container image" | **[D]** |
| 11 | `TRITON_CACHE_DIR` must be set **separately** | prior in-repo research (`scripts/serving/STARTUP.md` §1.6) citing `compilation/decorators.py:550-559` vs `compiler_interface.py:475-481`: on the AOT path (default with torch >= 2.10, and 0.28 pins torch 2.13.0) vLLM sets only `TORCHINDUCTOR_CACHE_DIR` | Triton's JIT cache otherwise dies with the container | **[R]** (not re-verified this session) |
| 12 | `OMP_NUM_THREADS = available_cpus // 8` | `vllm/utils/torch_utils.py:224-231`: `startup_omp_num_threads(n) = max(1, available_cpu_count() // max(1,n))`, where `available_cpu_count()` = `len(os.sched_getaffinity(0))` capped by the cgroup quota; called with `local_world_size` from `v1/executor/multiproc_executor.py:133,1105`. `set_multiprocessing_worker_envs` returns early if `OMP_NUM_THREADS` is already in the env, so an explicit value is respected verbatim | 244 CPUs / 8 -> **30**. At DP=1 vLLM already computes this, so an explicit value is belt-and-braces, not a fix | **[D]** |
| 13 | The DP oversubscription bug does **not** apply to you | vLLM issue #52330 (closed) fixed by #52385 "[Bugfix] Account for local DP workers in startup thread allocation", merged **2026-08-17**, i.e. before the 0.28.0 release (2026-08-26); and `available_cpu_count()` in the 0.28.0 tree is already cgroup-aware | the 1822 s -> 118 s (15.4x) figure in #52330 was DP=4; you run DP=1 | **[D]** code read; **[R]** the numbers |
| 14 | MXFP4 post-load repacking is real and per-expert | `quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py:149-211` | `process_weights_after_loading` runs `for e_idx in range(E): swizzle_mxfp4_scales(...)` then `torch.stack` — i.e. **896 iterations per MoE layer, x92 layers**, on GPU, single-threaded Python | **[D]** |
| 15 | There is no way to cache or skip #14 in 0.28.0 | `model_loader/base_loader.py:load_model` calls `process_weights_after_loading(...)` unconditionally for **every** loader, after `load_weights` | no flag, no on-disk artifact | **[D]** |
| 16 | Official image covers sm_103 | `CMakeLists.txt:118-129,465,1031,1097`; `cmake/utils.cmake:416-445` | Blackwell kernels are built as family-specific `10.0f`, which runs on 10.3 | **[D]** |
| 17 | `--max-cudagraph-capture-size` default on B300 is 1024 -> 83 shapes | `config/compilation.py:692-707`: "capped at 512 by default, or **1024 on data center Blackwell GPUs**"; list = `[1,2,4] + range(8,256,8) + range(256, max+1, 16)` | 3 + 31 + 49 = 83 sizes, x2 descriptor sets under the default `FULL_AND_PIECEWISE` | **[D]** |
| 18 | InstantTensor tuning knobs | https://raw.githubusercontent.com/scitix/InstantTensor/main/README.md | `INSTANTTENSOR_BACKEND` (default `[URING, AIO]` on disk, `MMAP` on tmpfs; options `AIO`,`URING`,`CUFILE`,`AIO_BUFFERED`,`URING_BUFFERED`,`MMAP`), `INSTANTTENSOR_CONCURRENCY`, `INSTANTTENSOR_IO_DEPTH` (max 1024), `INSTANTTENSOR_CHUNK_SIZE`, `INSTANTTENSOR_BUFFER_SIZE`, `INSTANTTENSOR_MAX_FREE_MEM_USAGE` (0.5), `INSTANTTENSOR_CACHE_BUFFER` (0), `INSTANTTENSOR_DEBUG` (0) | **[R]** |
| 19 | InstantTensor published numbers | `docs/models/extensions/instanttensor.md` | Qwen3-30B-A3B 1xH200 57.4 s -> 1.77 s (32.4x, 35 GB/s); DeepSeek-R1 8xH200 160 s -> 15.3 s (10.5x, 45 GB/s) | **[R]** (local NVMe, not a shared POSIX FS) |
| 20 | fastsafetensors cannot use GDS at TP>1 | `model_executor/model_loader/weight_utils.py:1054-1057`: `nogds = pg.size() > 1`, unconditional, "to avoid `cuFileDriverOpen()` which initializes the GDS DMA subsystem for all visible GPUs" | GDS is off for you no matter what weka supports; only knob is `VLLM_FASTSAFETENSORS_QUEUE_SIZE` (`envs.py:123`, default 0) | **[D]** |
| 21 | fastsafetensors stages a whole shard in device memory | vLLM PR #55985 (**open**, not in 0.28.0), "[Model Loader] Bound fastsafetensors peak memory with the 0.4 fit planner" | "it stages entire shards in device memory before distribution"; example OOM: 46.52 GiB shard on a card with 38.31 GiB free. K3's shards are 1561 GB / 96 = **~16.3 GB** and you have ~87 GiB free per GPU after weights, so this should fit — but it is the failure mode to watch | **[R]**, with an **[I]** headroom calculation |
| 22 | EP weight filter exists and matches K3's names | `model_loader/ep_weight_filter.py` (`_EXPERT_ID_RE = r"\.experts\.(\d+)\."`), gated in `default_loader.py:351-411` on `is_moe and enable_expert_parallel and enable_ep_weight_filter` (and not `enable_eplb`) | K3's `...block_sparse_moe.experts.895.w3.weight_packed` matches | **[D]** |
| 23 | ...but it skips only `.weight` / `.weight_packed`, never scales | `ep_weight_filter.py:should_skip_weight` — "Only skip heavy weight tensors, never scale/metadata tensors" | at EP=8 the per-rank *byte* count drops ~8x for experts but the per-rank *tensor* count only drops from ~495k to ~278k (1.8x), because all 247k `weight_scale` tensors are still read. **[I]** If your bottleneck is per-tensor cost, this buys much less than the byte figure suggests | **[D]** code; **[I]** the arithmetic |
| 24 | EP filter is incompatible with fastsafetensors / instanttensor | `default_loader.py:267-291`: `local_expert_ids` is passed only to `safetensors_weights_iterator`, not to the fastsafetensors or instanttensor branches | you must choose one | **[D]** |
| 25 | Loader choice does **not** invalidate the compile cache | `config/load.py:compute_hash` returns a hash of an empty factor list with the comment "this config will not affect the computation graph" | A/B loaders freely | **[D]** |
| 26 | `enable_flashinfer_autotune` / `enable_jit_warmup` / `enable_cutedsl_warmup` are compile-hash-neutral | `config/kernel.py:256-267` ignore list | you can flip these without a cold compile | **[D]** |
| 27 | K3 has its own Triton warmup, gated on `enable_jit_warmup` | `model_executor/warmup/kimi_k3_triton_warmup.py` + `kernel_warmup.py` | warms `attn_res` across block profiles, and `fused_recurrent_kda` **only if `num_spec > 0`** — so with no speculative decoding the KDA half is a no-op | **[D]** |
| 28 | K3 has no MTP head in this checkpoint | `config.json` `text_config.num_nextn_predict_layers: 0` | no drafter weight load, no second compile, no second capture set (unless you opt into the recipe's DSpark speculator, which is a separate model download) | **[D]** |
| 29 | Recipe's own startup-relevant env for other profiles | recipe `strategy_overrides.multi_node_tp_pp.extra_env` | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: 1800` alongside `VLLM_ENGINE_READY_TIMEOUT_S: 3600`, justified as "~1.68 TB of weights off shared storage" | **[D]** |

---

## 3. What invalidates the caches (freeze these before you start measuring)

**[R]** from the prior in-repo research in `scripts/serving/STARTUP.md` §1.2, which
read `envs.py:compile_factors()` and the per-config `compute_hash` methods; I
re-verified the two that matter most for this plan (`LoadConfig`, `KernelConfig`)
and both are hash-neutral **[D]**.

Re-keys the torch.compile cache: `--tensor-parallel-size`,
`--decode-context-parallel-size`, `--kv-cache-dtype`, `--max-model-len`,
`--max-num-batched-tokens`, `--enable-expert-parallel`, any cudagraph field,
`--moe-backend`, and **every `VLLM_*` var not on the explicit ignore list** —
including `VLLM_ENGINE_READY_TIMEOUT_S`, `VLLM_USE_DEEP_GEMM`,
`VLLM_ALLREDUCE_USE_FLASHINFER`, `VLLM_USE_V2_MODEL_RUNNER`.

Hash-neutral, safe to sweep: `--load-format`, `--safetensors-load-strategy` and its
thread/block knobs, `--gpu-memory-utilization`, `--kv-cache-memory-bytes`,
`--enable-flashinfer-autotune` / `--enable-jit-warmup` / `--enable-cutedsl-warmup`,
`VLLM_CACHE_ROOT`, `VLLM_ENABLE_STARTUP_PLAN`, `TRITON_CACHE_DIR`,
`FLASHINFER_WORKSPACE_BASE`, `DG_JIT_CACHE_DIR`, `INSTANTTENSOR_*`, `RUNAI_*`,
`OMP_NUM_THREADS`.

**[D] Verification flag:** `VLLM_FORCE_AOT_LOAD=1` makes the boot fail loudly
instead of silently recompiling on a cache miss
(`docs/configuration/optimization.md:23`).

---

## 4. Throughput risks, called out explicitly

Per your constraint, every suggestion that could cost decode throughput or KV
capacity:

1. **`--no-enable-flashinfer-autotune` — real, accepted, but real.**
   **[D]** With autotune off, FlashInfer "will rely on heuristics, which may be
   significantly slower" (docstring of `flashinfer_autotune` in `kernel_warmup.py`).
   **[D]** The official vLLM recipe sets it for *both* the Blackwell and Hopper K3
   profiles, and for the PD-cluster prefill role, so Moonshot/vLLM have accepted the
   trade for this model. **[I]** K3's routed experts go through the
   compressed-tensors MXFP4 MoE method (CUTLASS or Marlin), not through the
   FlashInfer FP4 GEMM that autotune mainly targets, which is a plausible reason the
   recipe is comfortable. **Measure decode tok/s with and without before you make it
   permanent** — it is hash-neutral so the A/B costs one boot, not one compile.

2. **`--gpu-memory-utilization 0.95` is the recipe value but raises OOM risk.**
   **[D]** The recipe's own `multi_node_tp_pp` block drops to 0.90 with the note
   that "the flashinfer TRTLLM MXFP4 MoE kernel allocates a ~1.6 GiB runtime
   workspace outside vLLM's pool on the first forward, and at 0.95 a 180 GB B200
   OOMs on the first warmup pass. B300 has the headroom to spare". You have B300, so
   0.95 is sanctioned — but note that `VLLM_ENABLE_STARTUP_PLAN` persists whatever
   number 0.95 profiled to, and an OOM on a later boot means deleting the plan.

3. **CUDA-graph trimming is a pure throughput trade and I am not recommending it as
   a default.** **[D]** Capture cannot be persisted (a CUDA graph holds device
   pointers in the live context; nothing in vLLM serialises one). The only lever is
   capturing less:
   - `--max-cudagraph-capture-size 256` cuts the list from 83 to 35 sizes (~58%
     fewer captures). **[I]** With `--max-num-seqs 64` and no speculative decoding,
     uniform-decode batches never exceed 64 tokens, so the *decode* graphs you
     actually replay are all still captured; what you lose is graphs for
     mixed/chunked-prefill batches above 256 tokens. For a short-prompt,
     128k-generation workload that is probably free — **probably**, not certainly.
   - `-cc.cudagraph_mode=FULL_DECODE_ONLY` halves the descriptor sets but makes
     mixed prefill-decode batches run without a graph. **I do not recommend this**
     for a run whose numbers you will publish.
   - `--enforce-eager` / `-O0`: fastest possible boot, largest decode loss
     (`docs/configuration/optimization.md:25`: "at the cost of steady-state decode
     performance"). **Use only for measuring how much of your boot is compile+capture.**
   Both cudagraph knobs **re-key the compile cache** (`CompilationConfig.compute_hash`
   hashes all fields), so pin one value per cache lineage.

4. **`--enable-expert-parallel` (needed for `--enable-ep-weight-filter`) changes how
   the MoE runs**, from TP-sharded experts to EP-routed experts, with all-to-all
   dispatch. That is a throughput decision, not a startup one. The recipe's
   `single_node_tp` default for B300 leaves EP **off**. Do not turn EP on to speed up
   loading.

5. **`--kv-cache-memory-bytes` set by hand caps concurrency** (`optimization.md:24`).
   Use `VLLM_ENABLE_STARTUP_PLAN=1` instead, which reproduces what a cold boot would
   have measured and self-invalidates.

---

## 5. Measurement plan (deterministic, one variable at a time)

You currently cannot attribute your 997–1398 s. Instrument first.

**Log lines to grep for, all [D] from the 0.28.0 tree:**

| Phase | Log line | Source |
|---|---|---|
| weight load | `Loading weights took %.2f seconds` | `default_loader.py:430` — note it is `logger.info_once`, so **you see one rank's number, not the spread** |
| EP filter engaged | `EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts` | `default_loader.py:405-411` |
| MXFP4 MoE backend chosen | `Using CutlassExpertsMxfp4 for MXFP4 MoE` / `Using MarlinExperts for MXFP4 MoE` | `compressed_tensors_moe_w4a4_mxfp4.py:52,60` |
| AOT compile hit | `Directly load AOT compilation from path ...` | `compilation/decorators.py` **[R]** from prior research |
| autotune cache | `Using FlashInfer autotune cache file: ...` / `Skipping FlashInfer autotune because it is disabled.` | `kernel_warmup.py` |
| startup plan | `Applying persisted startup plan (fingerprint %s): kv_cache_memory_bytes=%d ...` / `Saved startup plan to %s` | `startup_plan.py:131,178` |
| graph capture | `Graph capturing finished in %.0f secs, took %.2f GiB` | **[R]** prior research, `gpu_model_runner.py` |

**Suggested ladder (each step one variable, all hash-neutral unless noted):**

1. Boot once with the §1 config. Record the phase breakdown from the log lines above.
   This alone tells you whether your 1000 s is read, repack, JIT, or capture.
2. If weight load dominates: A/B `fastsafetensors` vs `instanttensor` vs
   `--safetensors-load-strategy eager` vs `prefetch --safetensors-prefetch-num-threads 32
   --safetensors-prefetch-block-size 67108864`. All four are compile-hash-neutral, so
   this is four boots against one compile cache.
3. If you keep instanttensor, set `INSTANTTENSOR_DEBUG=1` once and try
   `INSTANTTENSOR_BACKEND=AIO_BUFFERED` (or `MMAP`) with
   `INSTANTTENSOR_CONCURRENCY` / `INSTANTTENSOR_IO_DEPTH` raised. **[I]** The default
   `[URING, AIO]` is *direct* I/O, which on WEKAFS bypasses the client page cache;
   that is a plausible explanation for 167 MB/s effective against a 3 GB/s node and
   for the anti-correlation you saw.
4. Pre-warm the page cache synchronously before vLLM starts, as a separately timed
   phase you can alert on:
   `find $MODEL_DIR -name '*.safetensors' -print0 | xargs -0 -P 32 -I{} dd if={} of=/dev/null bs=64M status=none`.
   **[I]** With ~3 TB host RAM and a 1.56 TB checkpoint this fits, and it converts a
   random-read tail into a bandwidth-bound sequential pass.
5. Only after the load is understood, touch CUDA graphs — and then measure decode
   throughput, not just boot time.

---

## 6. Rejected options, and why

| Option | Verdict | Reason |
|---|---|---|
| `--load-format sharded_state` (pre-shard once for TP=8, skipping both the 500k-tensor parse *and* the MXFP4 repack) | **Rejected — it will not work** | **[D]** `base_loader.py:load_model` calls `process_weights_after_loading(...)` for *every* loader, including `ShardedStateLoader`. `save_model` dumps the post-processed `state_dict` (keys like `w13_weight`), but a fresh model built for load has `w13_weight_packed` (created by `CompressedTensorsW4A4Mxfp4MoEMethod.create_weights`), so the loader's `state_dict` lookup mismatches and `sharded_state_loader.py:161-162` raises `Missing keys ... in loaded state!`. This was the most attractive idea on paper; the code closes it. |
| `--load-format runai_streamer` / `runai_streamer_sharded` | **Rejected as a default** | **[R]** prior research found `runai_streamer_sharded` maps to `ShardedStateLoader` (same problem as above) and that `--model-loader-extra-config '{"concurrency":...}'` on it raises. Plain `runai_streamer` is a legitimate A/B (env vars `RUNAI_STREAMER_CONCURRENCY`, `RUNAI_STREAMER_MEMORY_LIMIT`, read by the C++ streamer, not by vLLM), but the recipe picked fastsafetensors for this model and RunAI's own docs scope distributed streaming to object storage, not POSIX. **[R]** The InstantTensor benchmark measured RunAI distributed at **0.83x** (slower) on a filesystem. |
| `fastsafetensors` with GPUDirect Storage | **Not available to you** | **[D]** `weight_utils.py:1054-1057` forces `nogds = pg.size() > 1` unconditionally at TP>1, regardless of whether WekaFS is GDS-qualified. Only patching that line would change it. |
| `--enable-expert-parallel --enable-ep-weight-filter` | **Rejected as a default; optional experiment** | **[D]** It works on K3's names, but (a) it is mutually exclusive with `fastsafetensors`/`instanttensor` (`default_loader.py:267-291`), (b) **[D]** it skips only `.weight_packed`, never `.weight_scale`, so per-rank tensor count falls only ~1.8x while bytes fall ~8x — and your evidence says tensor count is what hurts, and (c) it forces an EP MoE execution change that the recipe does *not* use for the single-node B300 profile. |
| `--enforce-eager` / `-O0` for production | **Rejected** | **[D]** `docs/configuration/optimization.md:12,25`. Decode-heavy MoE serving is exactly where per-op launch overhead hurts. Useful only as a diagnostic to quantify compile+capture. |
| `--kv-cache-memory-bytes <literal>` | **Rejected in favour of `VLLM_ENABLE_STARTUP_PLAN=1`** | **[D]** Same startup saving, but the plan is fingerprinted, free-memory-gated and self-invalidating (`startup_plan.py`), whereas a literal is "only valid on the same GPU with the same initial free memory" (`optimization.md:24`). |
| `VLLM_DEEP_GEMM_WARMUP=skip` | **Rejected** | **[D]** `envs.py` — `skip` moves the JIT cost to the first real request rather than removing it. Good for a time-to-ready metric, bad for first-token latency. Also **[R]** it re-keys the compile cache (not on the ignore list). |
| `--kernel-config.enable_cutedsl_warmup=false` | **Not recommended** | K3's NVIDIA path has real CuTe-DSL kernels (`vllm/models/kimi_k3/nvidia/ops/cute_dsl/{gemm_rs,latent_moe_tail}`), so skipping warmup moves their JIT into the first request. Hash-neutral, so it is a valid *diagnostic* to measure how big that phase is. |
| `vllm/vllm-openai:kimi-k3` image | **Rejected** | Pushed 2026-07-27, a month before the `v0.28.0` release that contains full K3 support. Use `v0.28.0-x86_64`. |
| `--moe-backend deep_gemm_mega_moe` | **Rejected** | **[D]** Recipe scopes it to GB200/GB300 NVL-tray DEP deployments only: "B200/B300 leave `--moe-backend` unset and use vLLM's default automatic selection". It also re-keys the compile cache (`KernelConfig.compute_hash` hashes `moe_backend`). |
| DSpark speculative decoding (recipe `spec_decoding` feature) | **Out of scope, and a startup cost** | It adds a second model (`RedHatAI/Kimi-K3-speculator.dspark`) to download, load and compile, and **[D]** turns on the `fused_recurrent_kda` half of `kimi_k3_triton_warmup` (gated on `num_spec > 0`). It is a large *throughput* win per the blog (**[R]** 111-118 -> 331-370 tok/s per user, 3.14x) — so it may well be worth the startup cost, but that is a throughput decision, not a startup one. |

---

## 7. MXFP4 post-load processing — the direct answer to question 6

**Yes, and it is not avoidable or cacheable in 0.28.0. [D]**

1. K3's `text_config.quantization_config` is `quant_method: compressed-tensors`,
   `format: mxfp4-pack-quantized`, with an `ignore` list covering `self_attn`,
   `shared_experts`, dense `mlp.(gate|up|gate_up|down)_proj`, `lm_head`,
   `vision_tower`, `mm_projector` — i.e. **only the 896 routed experts are MXFP4**.
2. That routes to `CompressedTensorsW4A4Mxfp4MoEMethod`
   (`.../compressed_tensors_moe_w4a4_mxfp4.py`). Its
   `process_weights_after_loading` (line 149):
   - rebinds `w13_weight_packed` -> `w13_weight` and `delattr`s the packed name;
   - **if** `CutlassExpertsMxfp4._supports_current_device()` (which calls
     `ops.mxfp4_experts_quant_supported(capability)`), runs a Python
     `for e_idx in range(E)` loop calling `swizzle_mxfp4_scales` twice per expert,
     then two `torch.stack`s — **896 iterations per MoE layer, 92 layers**;
   - **else** falls back to `prepare_moe_fp4_layer_for_marlin(layer)` with the
     warning "Your GPU does not have native support for FP4 computation ... This may
     degrade performance for compute-heavy workloads."
3. `base_loader.py:load_model` calls `process_weights_after_loading` unconditionally
   after `load_weights`, for every load format. There is no flag to skip it and no
   on-disk artifact of the result.
4. **[I]** On B300 you should land on the CUTLASS branch (log line
   `Using CutlassExpertsMxfp4 for MXFP4 MoE`). Confirm from the log — if you see
   `Using MarlinExperts for MXFP4 MoE` instead you are paying a much heavier repack
   *and* a slower steady-state kernel, and that is a bug worth reporting.
5. **[I]** This phase sits between "weights loaded" and the first compile log, and
   nothing times it explicitly. If your boot has a long silent gap there, this is it.
   The only mitigations available are structural (fewer experts per rank via EP —
   see §6) and none is free.

A separate, unrelated per-expert loop exists for the *other* MXFP4 path
(`fused_moe/oracle/mxfp4.py:765-870`, TRTLLM backends), which does six
gather/permute ops per expert with a shape-keyed `_cache_permute_indices`. **[D]**
That path asserts `w13_bias is not None`, which K3's checkpoint does not provide, so
you should not reach it.

---

## 8. Open questions / could not verify

1. **No published fastsafetensors-vs-instanttensor number for K3 on a shared POSIX
   FS.** The recipe says "much faster weight load" with no figure; the vLLM blog
   prints the flag without a timing. Every number I found for either loader is on
   local NVMe. **This is the single biggest unknown and only your cluster can settle it.**
2. **`recipes.vllm.ai/moonshotai/Kimi-K3` renders from
   `models/moonshotai/Kimi-K3.yaml` in `vllm-project/recipes` (fetched and quoted
   above), but that YAML says `min_vllm_version: "0.27.1"` and its `install.docker`
   note still reads "Use a K3-enabled nightly image after the integration lands" —
   i.e. the page is partly stale relative to the 0.28.0 release.** The YAML also
   describes the model as **2.8T** params / `vram_minimum_gb: 1680` "pre-release
   estimate ... Replace with the real safetensors footprint once weights are
   published", whereas the published checkpoint's `index.json` gives
   **1,560,860,324,864 B** and you report 5.46T total params. Treat the recipe's
   *flags* as authoritative and its *numbers* as pre-release estimates.
3. **`docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3` could not be read
   usefully.** The page is an interactive React deployment configurator, not prose;
   the fetch returned component scaffolding and flag *categories* (`--tp-size`,
   `--dp-size`, `--ep-size`, `--moe-a2a-backend`, `--moe-runner-backend`,
   `--enable-hierarchical-cache`, PD disaggregation, "MXFP4 megamoe") but **no
   complete launch command and nothing about weight loading or startup**. I did not
   find the underlying data file. If SGLang startup guidance matters, that page needs
   a browser, not a fetch.
4. **`TRITON_CACHE_DIR` gap (evidence row 11)** is carried over from the prior
   in-repo research and I did not re-open `compilation/decorators.py` this session.
   It is cheap and harmless to set regardless.
5. **Whether `--trust-remote-code` is actually required.** vLLM bundles
   `KimiK3Config`, the processor (`transformers_utils/processors/kimi_k3.py`) and the
   tokenizer mode, so it may be droppable — which would remove an HF round-trip and
   an `exec` of downloaded Python per rank. The recipe keeps it. I did not trace the
   processor/tokenizer load path far enough to be sure.
6. **Whether `CutlassExpertsMxfp4._supports_current_device()` returns True on
   sm_103.** It delegates to the C++ `ops.mxfp4_experts_quant_supported(capability)`
   (`csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu`), which I did not
   read. The log line settles it in one boot.
7. **Exact fastsafetensors peak-memory behaviour at 96 shards x 16.3 GB with
   `nogds=True` and TP=8.** PR #55985 describes shard-sized device staging on the GDS
   path; I did not confirm the buffered path's peak. My headroom arithmetic says it
   fits with ~87 GiB free per GPU, but that is **[I]**.
8. **`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`, `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM`,
   `VLLM_KIMI_K3_GEMM_RS`** all exist (`envs.py:209-211`, all default `0`). The
   recipe sets the first only in the PD-cluster prefill role. I did not evaluate
   their startup or throughput effects.
9. **vLLM issue #50587 "Kimi K3 Performance Optimization"** is an open tracking issue
   with many kernel wins, but I found **nothing** in it about startup time, weight
   loading, MXFP4 post-processing, CUDA-graph capture or warmup. Issue #49349 "Zero
   JIT compilation during runtime" tracks moving *all* JIT into warmup (including a
   Kimi-K3 migration) and proposes a `--jit-monitor-mode error` flag — **I did not
   verify that flag exists in 0.28.0**.

---

## 9. Relationship to `scripts/serving/STARTUP.md`

That file is a prior research pass covering Qwen3.5 / Kimi-K2.6 / DeepSeek-V3.2 /
GLM-5.2 on the same hardware. Its §1 (compile-cache keying), §2 (FlashInfer/DeepGEMM
JIT caches), §3 (CUDA graphs) and §4 (weight loading) apply to K3 essentially
unchanged and are not repeated here. Two corrections this pass produced:

- **[D]** The `OMP_NUM_THREADS` oversubscription bug it describes (issue #52330) was
  fixed by PR #52385, merged 2026-08-17, **before** the 0.28.0 release — and the
  0.28.0 `available_cpu_count()` is cgroup-aware. It is not an active bug for a
  DP=1 TP=8 job. The *advice* to set the value explicitly still stands.
- **[D]** Its §4.5 recommendation of `sharded_state` as a TP-locked fast path does
  not work for a compressed-tensors MXFP4 checkpoint (see §6 above).

---

## 10. Crash investigation

Research pass of 2026-09-16, against the three mid-generation CUDA faults on
`vllm/vllm-openai:v0.28.0-x86_64`, 8x B300 (sm_103). Claim markers as elsewhere in
this file: **[D]** read in code or in an issue/PR I opened, **[R]** third-party report,
**[I]** my inference.

### 10.1 The one-paragraph answer

The prime suspect is **not** KDA and **not** `TOKENSPEED_MLA`. It is the
**`--decode-context-parallel-size 8 --dcp-comm-backend a2a` subsystem**, and
specifically the *direct symmetric-memory* implementation of the DCP collectives,
which **0.28.0 turns on automatically and silently** whenever the DCP group spans
NVLink — as it does on a single B300 node. That path is a set of hand-written CUDA
kernels that write through raw peer pointers with `multimem.*` PTX and synchronise
with a **device-side spin-wait that ends in `asm volatile("trap;")`**. It is gated by
three env vars that default to "auto", it has **no layout or capacity fallback in
0.28.0** (0.29.0 added one), and it is byte-identical between 0.28.0 and 0.29.0. The
cheapest high-yield experiment is to disable it — three env vars, no serve-flag
change, no re-plan of the KV cache. The observed pre-crash signature (power 620 W ->
240 W, utilisation pinned at 100%, throughput 0, requests still `Running`) is the
signature of a **device-side spin loop**, which is what that code does and what
almost nothing else in the decode path does **[I]**.

Secondarily: this box has **up to three independent NVLink-multicast consumers live
at once** — the direct DCP kernels, `SymmMemCommunicator` (`VLLM_ALLREDUCE_USE_SYMM_MEM`,
**defaults to 1** and you have not disabled it), and FlashInfer MNNVL all-reduce
(Attempt A only). The nearest precedent on this exact hardware, issue #50147, was an
IMA on 8x B300 Kimi-K3 whose coredump named a **multicast all-reduce kernel**.

### 10.2 Ranked remediation list

Ranked by (expected yield / cost). Items 0 and 1 are compatible with each other and
with everything below; run 0 always.

---

**0. Capture a CUDA coredump. [diagnostic, ~zero cost — do this on the next run
regardless of what else you change]**

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
CUDA_COREDUMP_SHOW_PROGRESS=1
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
CUDA_COREDUMP_FILE="/weka/.../cuda_coredump_%h.%p.%t"
```

- **Cost:** negligible at runtime; a few hundred MB of dump per crash. Point
  `CUDA_COREDUMP_FILE` at persistent storage, not container-local disk.
- **Evidence:** **[R]** this is precisely what resolved #50147 — the same model on the
  same 8x B300 hardware, crashing every 11-20 min with a *different reported site each
  time*. Three engine-killing crashes and a pile of tracebacks got nowhere; one
  coredump named `flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16,
  QuantType::kNone, false, 1, float4>` and the issue was closed four days later. The
  procedure is vLLM's own, from https://vllm.ai/blog/2025-08-11-cuda-debugging .
- **Why it matters here:** all three of your faults are *asynchronous* reports. The
  `~CUDAEvent` warning, the `CachingHostAllocator` pinned-free exception and the
  `cuMemFree` in `SymmDeviceMemory.__del__` are all **sticky-error teardown noise from
  an already-poisoned context**, not the fault site **[I]**. You cannot rank
  hypotheses further without naming the kernel.
- **Also grep the next log for these three lines**, all of which are `info_once` in
  0.28.0 (`vllm/v1/attention/ops/dcp_utils.py:655,691,726`) **[D]**:
  `Using direct symmetric-memory DCP A2A for MLA.` /
  `... DCP query gather for MLA.` / `... chunked-context KV gather for MLA.`
  Their presence confirms hypothesis #1 is live in your build; their absence kills it.
  Also grep for `direct DCP A2A timeout` / `direct DCP q-gather multimem timeout` —
  those are `printf`s emitted from the device immediately before the `trap` **[D]**.
- **Confidence:** n/a (diagnostic). This is the highest-information action available.

---

**1. Disable the direct symmetric-memory DCP kernels. [highest yield per unit cost]**

```bash
VLLM_USE_DIRECT_DCP_A2A=0
VLLM_USE_DIRECT_DCP_Q_GATHER=0
VLLM_USE_DIRECT_DCP_KV_GATHER=0
```

- **Throughput cost:** low — low single-digit percent of decode **[I]**. These are a
  latency optimisation over the generic NCCL/Triton A2A combine, not a capability.
  For calibration, the closely-related #55289 measured the *masking chain alone* at
  1.23% of end-to-end decode throughput **[D]**. You keep DCP, keep `a2a`, keep
  `TOKENSPEED_MLA`, keep the KV-cache layout; only the collective implementation
  changes. No restart cost beyond the restart itself.
- **Evidence:**
  - **[D]** The three vars exist in 0.28.0 at `vllm/envs.py:199-201` typed
    `bool | None = None`, parsed at `envs.py:2113-2121` via `maybe_convert_bool`, with
    the in-source comment *"Direct DCP ops default on when applicable; set to 1 to
    enforce or 0 to disable."* Auto-selection is `_direct_dcp_enabled` /
    `_direct_dcp_multicast_enabled` (`dcp_utils.py:67-94`): with the var unset it
    returns true when symmetric memory is available and `_symm_mem_spans_group()`
    succeeds. On a single 8-GPU NVLink node that probe succeeds, so **you are on this
    path today without having asked for it**.
  - **[D]** In 0.28.0 `MLADCPManager._init_combine` (`dcp_utils.py:635-665`) binds
    `functools.partial(direct_workspace.lse_reduce, ...)` **unconditionally** once the
    workspace exists — there is no capacity check and no layout check. In 0.29.0 the
    same method binds a new wrapper `_direct_workspace_combine` (`v1/attention/ops/dcp.py:1303-1328`)
    that falls back to `dcp_a2a_lse_reduce` when `partial_output.shape[0] >
    direct_workspace.max_num_tokens`, with the comment *"Forced MQA path pass all batch
    tokens (including prefill) into combine, which may exceed the direct
    symmetric-memory workspace."* **That fallback does not exist in 0.28.0.**
  - **[D]** The kernels themselves: `csrc/libtorch_stable/attention/dcp_utils/`.
    `dcp_direct_common.cuh` defines `get_peer_ptr` (a raw `int64 -> T*` reinterpret of a
    peer address), `multimem_store_16` (`multimem.st.relaxed.sys.global.v4.f32`, a
    16-byte-aligned multicast store), and `wait_for_epoch`, which spins up to
    `kSpinLimit = 100000000` on `ld.global.acquire.sys.u32`. Every call site — 
    `dcp_direct_a2a_lse_reduce.cu:158-166`, `dcp_direct_q_gather.cu:73-77` — responds to
    a spin timeout with a `printf` followed by `asm volatile("trap;")`, which kills the
    context.
  - **[D]** Synchronisation is a two-slot double buffer keyed on `epoch & 1`
    (`parity`/`buffer_slot`). If any rank's epoch skews by two relative to a peer, a
    slot is reused before the peer has drained it. Nothing in the file enforces a
    global barrier between layers.
  - **[R]** Issue #54305 (open) — *"Direct DCP A2A crashes on GLM sparse-MLA strided
    output"*: the reporter confirms auto-selection with `VLLM_USE_DIRECT_DCP_A2A` unset,
    confirms *"the direct path calls the C++ kernel unconditionally"*, and lists
    `VLLM_USE_DIRECT_DCP_A2A=0` as workaround 1 of 2, validated end to end.
- **Honest caveat:** #54305's own failure mode is a **clean `RuntimeError` at
  CUDA-graph capture**, not a mid-run IMA, and it is a different attention backend.
  It establishes *that the path is auto-on and unguarded*; it does **not** establish
  that it is your fault site. The IMA argument is the `trap` + spin-loop mechanism and
  the power/utilisation signature, and that part is **[I]**.
- **Confidence:** **High** that this is the right first experiment. **Medium** that it
  is the root cause.

---

**2. Drop DCP entirely — fall back to the recipe's Blackwell baseline. [the known-good
configuration]**

Remove `--decode-context-parallel-size 8`, `--dcp-comm-backend a2a`, and change
`--attention_config.mla_prefill_backend` from `TRTLLM_RAGGED` back to `TOKENSPEED_MLA`.
Keep `--attention-backend TOKENSPEED_MLA`, `--kv-cache-dtype fp8` and
`--attention_config.use_prefill_query_quantization=true`.

- **Throughput cost:** real and workload-specific. DCP exists to shard the decode KV
  cache across ranks for exactly your profile (decode-heavy, long context), so this is
  the most expensive item on the list. I will not invent a number — measure it. **[I]**
- **Evidence:**
  - **[D]** In `recipes/models/moonshotai/Kimi-K3.yaml`, DCP is `features.text_only`,
    an **opt-in** block, not part of `hardware_overrides.blackwell`. The Blackwell
    baseline is `--kv-cache-dtype fp8`, `--attention-backend TOKENSPEED_MLA`,
    `--attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TOKENSPEED_MLA"}'`,
    `--enable-prefix-caching`, `--prefix-match-unit 128`, `--load-format fastsafetensors`,
    `--no-enable-flashinfer-autotune`. Note that **`TRTLLM_RAGGED` reaches your command
    line only via the DCP block** — the baseline prefill backend is `TOKENSPEED_MLA`.
    Dropping DCP therefore also drops `TRTLLM_RAGGED`, addressing item 5 for free.
  - **[R]** #41623 (open since 2026-05, last touched 2026-09-02) — Kimi-K2.6,
    `--tensor-parallel-size 8 --decode-context-parallel-size 8`, prefix caching,
    262144 context: *"Decode Context Parallelism produces unrelated gibberish output in
    latest nightly. This is a regression."* Your exact TP/DCP geometry and a sibling
    model. Still open, no fix merged.
  - **[R]** #54300 (open) — *"GlmMoeDsa (GLM-5.3) + decode-context-parallel: crashes on
    0.28.0, silently returns random tokens on 0.29.0"*, on 8x B200. A DCP-conditional
    regression whose crash half lands squarely on 0.28.0.
  - **[D]** #55780 (merged 2026-09-08, **not** in 0.28.0) flipped
    `AttentionImplBase.supports_dcp` to default `False` after finding that *"several
    unsupported implementations also inherited `True` and failed the worker's
    missing-LSE check after loading weights."* `TokenspeedMLAImpl` is on the explicit
    opt-in list, so **your backend genuinely supports DCP** — but the PR is evidence
    that DCP support was being advertised implicitly and inconsistently in the 0.28.0
    era.
- **Confidence:** **High** that this removes the suspect subsystem. **High** that it is
  a serviceable configuration (it is the recipe's own baseline).

---

**3. Disable the symmetric-memory all-reduce. [cheap, addresses the #50147 precedent]**

```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0     # you have NOT set this; it defaults to 1
VLLM_ALLREDUCE_USE_FLASHINFER=0   # you already set this
```

- **Throughput cost:** low-to-moderate. Falls back to vLLM's custom all-reduce / NCCL.
  Affects small all-reduces at TP8, which at 92 layers is not nothing. **[I]**
- **Evidence:**
  - **[D]** `vllm/envs.py:260-261` — `VLLM_ALLREDUCE_USE_SYMM_MEM: bool = True`
    (default `"1"` at `envs.py:1860-1862`) versus `VLLM_ALLREDUCE_USE_FLASHINFER: bool
    = False` (default `"0"` at `envs.py:1864-1866`). **So in 0.28.0 the FlashInfer
    all-reduce is off by default and the symmetric-memory one is on** — setting
    `VLLM_ALLREDUCE_USE_FLASHINFER=0` between Attempt A and Attempt B changed less than
    it looks like it did, because a second multicast all-reduce stayed live the whole
    time. (This also corrects the recipe's Blackwell block, which sets
    `VLLM_ALLREDUCE_USE_FLASHINFER: "1"`; you are deviating from the recipe here.)
  - **[D]** `device_communicators/symm_mem.py` — `SymmMemCommunicator` calls
    `torch.ops.symm_mem.multimem_all_reduce_` and its own failure message names
    `VLLM_ALLREDUCE_USE_SYMM_MEM=0` as the escape hatch. sm_103 is an explicitly tuned
    entry: `all_reduce_utils.py:71-76`, `"10.3": {2: 4 MiB, 4: 32 MiB, 6: 32 MiB,
    8: 64 MiB}`. So this is a supported, deliberately enabled path on your card — not
    an accident — but it is multicast, and it is the same family as #50147's fault.
  - **[R]** #50147 (closed 2026-08-03) — *"Kimi-K3 (TP=8, prefix caching): recurring
    illegal-memory-access crashes under concurrent load"* on **8x NVIDIA B300 SXM6**,
    `VLLM_USE_V2_MODEL_RUNNER=1`, crashes at ~20 min and ~11 min, *"the reported crash
    site differs each time"*, *"time-to-crash shrinks monotonically as concurrency
    grows"*. Reported sites included `buildNdTmaDescriptor` (FlashInfer TRT-LLM MLA
    decode) and `_causal_conv1d_fwd_kernel` (KDA conv state) — i.e. **the same
    scattered-victim pattern you are seeing**. Root cause per coredump: the MNNVL
    all-reduce `rmsNormLamport` kernel. Fixed by #50386, merged 2026-07-30, **which is
    already in your 0.28.0** — so #50147 is not your bug, but it is the strongest
    available evidence that on this exact hardware and model, multicast collectives are
    where IMAs come from.
- **Confidence:** **Medium.** The specific #50147 fault is fixed in your build; this is
  betting on the subsystem, not on a named defect.

---

**4. Keep `--enable-prefix-caching` off. [already done, free, and now better
justified than when you did it]**

- **Throughput cost:** zero for you — the measured hit rate was 0.0%.
- **Evidence:**
  - **[R]** #50147's title and body single out prefix caching, and note that enabling
    it *"forces mamba cache mode `align`"* on this hybrid model.
  - **[D]** It also pre-empts a 0.29.0 landmine. `_store_cache_checkpoints_kernel` —
    the KDA prefix-checkpoint store — **does not exist at all in 0.28.0** (I diffed
    `vllm/models/kimi_k3/nvidia/kda.py` between the two tags; it is added at
    `kda.py:238` in 0.29.0 by #53614, and reached only when `checkpoint is not None`,
    i.e. with partial prefix caching on the `flashkda` prefill backend). If you later
    upgrade *and* re-enable prefix caching, you walk into #55924 — see §10.4.
- **Confidence:** **High** that it is free. **Low-to-medium** that it was the cause,
  since your 0.0% hit rate means the code path was barely exercised. Do not expect
  this alone to have fixed the crash.

---

**5. Move `mla_prefill_backend` off `TRTLLM_RAGGED`.**

`--attention_config.mla_prefill_backend=TOKENSPEED_MLA` (or `FLASHINFER`). Subsumed by
item 2; list it separately only if you keep DCP.

- **Throughput cost:** low; a different prefill kernel. Prefill is a small share of
  your workload (1000 prompts, up to 128k output tokens each). **[I]**
- **Evidence:** **[D]** The recipe's own guide note says the three registered MLA
  prefill backends — `FLASHINFER`, `TRTLLM_RAGGED`, `TOKENSPEED_MLA` — are
  interchangeable with `use_prefill_query_quantization`, and that the Blackwell profile
  emits `TOKENSPEED_MLA`. **[R]** #54300's reporter found that under DCP,
  prefill-shaped batches routed through `TRTLLM_RAGGED` came back corrupt while
  decode-shaped batches were correct.
- **Confidence:** **Low-medium** on its own; **free** to fold into item 2.

---

**6. `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`. [diagnostic, expensive]**

- **Throughput cost:** material — you lose full-graph decode. 10-30% of decode
  throughput is the usual order **[I]**; also a shorter startup, since fewer graphs are
  captured.
- **Evidence:** **[R]** #52225 (which you found) recommends exactly this as an
  isolation step. **[R]** #45487 (merged before 0.28.0) — *"Fix IMA in DCP a2a decode
  under full CUDA graphs"*, repro `Kimi-K2.5-NVFP4`, DCP4, `a2a`, `--kv-cache-dtype
  fp8`, full CUDA graphs: the A2A staging buffers came from a growable workspace that
  was regrown after capture, invalidating addresses baked into captured graphs. The
  original defect is fixed in your build, but it establishes the failure family
  *a2a staging-buffer lifetime vs. graph-captured pointers*, and the direct DCP
  workspace is a **persistent symmetric allocation shared by every MLA layer** with
  device-side epoch state that graph replay cannot re-initialise **[I]**.
- **Confidence:** **Medium** as a discriminator (it will tell you whether graphs are
  involved), **low** as a fix you would want to keep.

---

**7. `--enforce-eager`. [last-resort discriminator]**

- **Throughput cost:** severe — do not run a 1000x8-sample job like this. Use it for a
  short reproduction only.
- **Evidence:** **[R]** #54649 explicitly notes *"Eager mode avoids the graph failure
  but is not performance-representative."*
- **Confidence:** **High** as a discriminator, **not a deployment option.**

---

**8. Lower concurrency further / cap `--max-num-seqs`.**

You went 256 -> 128 and still crashed, so this is mitigation, not a fix.

- **Evidence:** **[R]** #50147: *"Time-to-crash shrinks monotonically as concurrency
  grows — single-request testing will not reproduce this in reasonable time."*
- **Confidence:** **[I]** It buys uptime, it does not remove the defect. Worth knowing
  for planning a reproduction: to *reproduce* fast, raise concurrency.

---

**9. Upgrade to a nightly at or after `bfb443a6b6` (2026-09-08T21:07Z) — not to the
0.29.0 tag.** See §10.4; this is deliberately last.

---

### 10.3 Evidence table

| # | Item | URL | One-line | Applies? | Mark |
|---|---|---|---|---|---|
| E1 | `VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER}` | `vllm/envs.py:199-201`, `:2112-2121` @ v0.28.0 | Typed `bool \| None = None`; comment: "Direct DCP ops default on when applicable; set to 1 to enforce or 0 to disable" | **Yes — auto-on for you** | **[D]** |
| E2 | `_direct_dcp_enabled` / `_symm_mem_spans_group` | `vllm/v1/attention/ops/dcp_utils.py:40-94` @ v0.28.0 | With the env unset, direct path selected whenever symmetric memory spans the DCP group — true on a single NVLink node | **Yes** | **[D]** |
| E3 | No fallback in 0.28.0 `_init_combine` | `dcp_utils.py:635-665` @ v0.28.0 vs `v1/attention/ops/dcp.py:1303-1328` @ v0.29.0 | 0.29.0 added `_direct_workspace_combine` with a capacity fallback to `dcp_a2a_lse_reduce`; 0.28.0 binds the direct kernel unconditionally | **Yes — genuine 0.28.0 defect** | **[D]** |
| E4 | Spin-then-`trap` in the direct kernels | `csrc/libtorch_stable/attention/dcp_utils/dcp_direct_common.cuh:74-81`; `dcp_direct_a2a_lse_reduce.cu:158-166`; `dcp_direct_q_gather.cu:73-77` @ v0.28.0 | `wait_for_epoch` spins 1e8 times on `ld.global.acquire.sys.u32`, then `printf` + `asm volatile("trap;")` | **Yes — matches the 100%-util / low-power stall** | **[D]** mechanism, **[I]** attribution |
| E5 | Direct DCP kernels unchanged 0.28.0 -> 0.29.0 | diff of all four files at both tags | **Byte-identical.** Upgrading does not touch the prime suspect | **Yes** | **[D]** |
| E6 | #54305 Direct DCP A2A crashes on strided output | https://github.com/vllm-project/vllm/issues/54305 | Open. Direct path auto-selected and called with no layout gate; `VLLM_USE_DIRECT_DCP_A2A=0` is the accepted workaround | Partly — different backend, and its symptom is a clean `RuntimeError` at capture | **[R]** |
| E7 | #50147 K3 IMA on 8x B300 | https://github.com/vllm-project/vllm/issues/50147 | Closed 2026-08-03. Same model + hardware + V2 runner; crashes at 11-20 min; different site each time; coredump named `flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<...,float4>`; fixed by #50386 | **Precedent only** — #50386 merged 2026-07-30, already in 0.28.0 | **[R]** + **[D]** on the merge date |
| E8 | #50386 stale latent-MoE residual pointer in CUDA graphs | https://github.com/vllm-project/vllm/pull/50386 | Merged 2026-07-30, i.e. **in** 0.28.0 | Ruled out | **[D]** |
| E9 | `VLLM_ALLREDUCE_USE_SYMM_MEM` defaults to 1 | `vllm/envs.py:260`, `:1860-1862`; `device_communicators/symm_mem.py`; `all_reduce_utils.py:71-76` | `multimem_all_reduce_`, live on your box; sm_103 TP8 threshold 64 MiB. `VLLM_ALLREDUCE_USE_FLASHINFER` defaults to **0**, not 1 | **Yes — an unexamined multicast path** | **[D]** |
| E10 | #41623 DCP gibberish, TP8/DCP8 | https://github.com/vllm-project/vllm/issues/41623 | Open since 2026-05. Kimi-K2.6, TP8 + DCP8 + prefix caching, 262144 ctx: DCP alone produces unrelated gibberish. No fix | **Yes — same geometry, sibling model** | **[R]** |
| E11 | #54300 GLM + DCP regression on 0.28.0 | https://github.com/vllm-project/vllm/issues/54300 | Open. Crashes on 0.28.0, silent garbage on 0.29.0, 8x B200, DCP. Reporter isolates `TRTLLM_RAGGED` prefill under DCP as corrupt | **Partly** — different model family | **[R]** |
| E12 | #55780 Require explicit DCP support | https://github.com/vllm-project/vllm/pull/55780 | Merged 2026-09-08 (**not** in 0.28.0). Defaults `supports_dcp=False`; `TokenspeedMLAImpl` **is** an explicit opt-in | Confirms your backend is legitimately DCP-capable | **[D]** |
| E13 | #55924 KDA IMA int32 overflow | https://github.com/vllm-project/vllm/pull/55924 | Merged 2026-09-08T21:07Z. `state_stride_0` up to 516096 x `state_idx` 4000+ overflows int32 in `_store_cache_checkpoints_kernel` | **Ruled out for 0.28.0** — the kernel does not exist there; and it is **still unfixed at the v0.29.0 tag** (verified `kda.py:263`) | **[D]** |
| E14 | #53614 K3 internal prefix checkpoints | https://github.com/vllm-project/vllm/pull/53614 | Merged 2026-09-06; introduces the kernel E13 fixes, gated on prefix caching + `flashkda` prefill | Not in 0.28.0 | **[D]** |
| E15 | #54649 Kimi-K3 DSpark/DCP IMA | https://github.com/vllm-project/vllm/issues/54649 | Closed 2026-09-04. `MLAAttentionSpec.merge()` used an `assert` for target/draft separation; optimized-Python containers strip asserts; `set.pop()` then picks wrong metadata -> CUDA-graph IMA. "Both TokenSpeed MLA and FlashInfer MLA reproduce it" | **Ruled out — requires DSpark spec decode** | **[R]** |
| E16 | #55234 fix for E15 | https://github.com/vllm-project/vllm/pull/55234 | Merged 2026-09-04, so **is** in 0.29.0 | n/a (you have no spec decode) | **[D]** |
| E17 | #51313 K3 fp8 KV gating | https://github.com/vllm-project/vllm/issues/51313 | Open. `backend_supports_prefill_query_quantization()` requires Blackwell **and** backend in `{FLASHINFER, TRTLLM_RAGGED, TOKENSPEED_MLA}` | Confirms **your fp8 + prefill-quant + backend combination is the sanctioned one.** Also documents `--kv-cache-dtype fp8_ds_mla` as the bf16-prefill-query alternative | **[D]** |
| E18 | #45487 IMA in DCP a2a under full CUDA graphs | https://github.com/vllm-project/vllm/pull/45487 | Merged before 0.28.0. Kimi-K2.5, DCP4, `a2a`, fp8 KV, full graphs: growable A2A staging buffers regrown after capture | Fixed in your build; establishes the failure family | **[R]** |
| E19 | #55289 / #54889 A2A empty-shard masking | https://github.com/vllm-project/vllm/pull/55289 | Merged 2026-09-04. "Under DCP... a rank whose local shard is empty produces **undefined output and LSE**"; masking drives those rows' LSE to `-inf` | Perf refactor, not a fix; documents that empty DCP shards are a live correctness hazard | **[D]** |
| E20 | #54111 fused groupwise RMSNorm quant race | https://github.com/vllm-project/vllm/pull/54111 | Merged 2026-08-28 -> **in 0.29.0, not in 0.28.0.** Shared-memory race | Possible, unquantified | **[R]** |
| E21 | #50729 Mamba overlapping state copy race | https://github.com/vllm-project/vllm/pull/50729 | Merged **2026-08-17** -> already in 0.28.0 | Ruled out as an upgrade motive | **[D]** |
| E22 | #53000 MNNVL Lamport mailbox fix | https://github.com/vllm-project/vllm/pull/53000 | Merged **2026-08-24** -> already in 0.28.0 | Ruled out as an upgrade motive | **[D]** |
| E23 | #52998 FlashInfer all-reduce by default | https://github.com/vllm-project/vllm/pull/52998 | Merged **2026-08-20**, before 0.28.0 — yet `envs.py` at v0.28.0 still defaults `VLLM_ALLREDUCE_USE_FLASHINFER` to `0` | Nothing changes at upgrade | **[D]** |
| E24 | Kimi-K3 recipe | https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml | DCP is `features.text_only` (opt-in); Blackwell baseline prefill backend is `TOKENSPEED_MLA`, not `TRTLLM_RAGGED`; documents the `VLLM_USE_DIRECT_DCP_*` knobs as "default to auto" | **Yes** | **[D]** |
| E25 | DCP workspace sizing | `dcp_utils.py:97-116` @ v0.28.0 | `min(max_num_batched_tokens, max(max_num_seqs * tokens_per_seq, max_cudagraph_capture_size))`. For you: no spec decode, `max_num_seqs=128`, Blackwell default capture size 1024 -> **1024 tokens** | See §10.5 — candidate explanation for Attempt B | **[D]** sizing, **[I]** attribution |

### 10.4 Would 0.29.0 fix it? No — and the tag is a bad target

Verified against both tags directly, not against the changelog:

- **[D]** All four direct-DCP CUDA files are **byte-identical** between v0.28.0 and
  v0.29.0. The prime suspect is untouched.
- **[D]** 0.29.0 *does* add the capacity fallback of E3, which is a real robustness win
  for the DCP combine and would convert a hard failure into a slow path.
- **[D]** #50729, #53000, #52998 and #50386 — all four of the "that sounds relevant"
  fixes — merged **before** 2026-08-26 and are therefore **already in your 0.28.0**.
  The only post-0.28.0 memory-safety fix I found touching a subsystem you use is
  #54111 (RMSNorm quantization shared-memory race).
- **[D]** 0.29.0 *introduces a new KDA illegal-memory-access bug that 0.28.0 does not
  have.* #53614 adds `_store_cache_checkpoints_kernel`; #55924 fixes an int32 address
  overflow in it; #55924 merged 2026-09-08T21:07Z and the 0.29.0 release cut is
  2026-09-09T08:54Z — but I checked `kda.py:263` at the **v0.29.0 tag** and the fix is
  **not** there. Reached only with prefix caching on the `flashkda` prefill backend.
- **[R]** #54300 reports GLM + DCP as *"crashes on 0.28.0, silently returns random
  tokens on 0.29.0"* — a reminder that on the DCP path, 0.29.0 can trade a crash for
  silent corruption, which for a 1000x8 generation job is strictly worse.
- **[D]** #53183 makes Model Runner V2 the default for all models in 0.29.0. You
  already force it with `VLLM_USE_V2_MODEL_RUNNER=1`, so this is neutral for you, but
  the 0.28.0 -> 0.29.0 range is 607 commits — a large confounder to introduce
  mid-investigation.

**If you upgrade, target a nightly at or after `bfb443a6b6` (2026-09-08T21:07Z)**, which
carries #55924 and #55234, rather than the v0.29.0 tag. But upgrading is item 9 for a
reason: it changes 607 commits at once and does not touch the prime suspect.

### 10.5 On the three specific attempts

- **Attempt A** (`VLLM_ALLREDUCE_USE_FLASHINFER=1`, misaligned address in
  `mnnvl.py:836 SymmDeviceMemory.__del__ -> cuMemFree(signal_pads_dev)`): a `cuMemFree`
  in a `__del__` cannot itself be the origin. Once a context takes a fault, every
  subsequent driver call returns the sticky error, and Python teardown is simply where
  the next call happens to be. Read this as *"something poisoned the context; the
  wreckage surfaced in FlashInfer's symmetric-memory teardown"* **[I]**. It does tell
  you a fabric/multicast allocation was live, which all three of E1/E9 and FlashInfer
  satisfy.
- **Attempt B** (HTTP 500s, then engine death, root cause lost): there is a concrete
  0.28.0 mechanism that produces exactly this shape. The direct A2A workspace is sized
  to **1024 tokens** for your config (E25), and `direct_dcp_a2a_lse_reduce` enforces
  `STD_TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens)` host-side
  (`dcp_direct_a2a_lse_reduce.cu:254`). In 0.28.0 there is **no fallback** when that
  bound is exceeded (E3), so a combine call carrying prefill-shaped token counts raises
  a `RuntimeError` per request — HTTP 500s — until the engine gives up. 0.29.0's new
  wrapper exists precisely to catch this case, and its comment names the trigger as the
  forced-MQA path passing prefill tokens into combine. **[D]** on the mechanism,
  **[I]** that it is what you hit — the log that would confirm it scrolled away.
  If you reproduce Attempt B, grep for `num tokens` / `max_num_tokens` in the traceback.
- **Attempt C** (`~CUDAEvent` warning at rank 4, `EngineDeadError`,
  `c10::AcceleratorError`, pinned-allocator rethrow): all four lines are sticky-error
  teardown. The single informative detail is **rank 4** — a *specific* rank faulting
  while others report only the collective aftermath is what you would expect from a
  peer-to-peer collective in which one rank's spin-wait expired **[I]**.

### 10.6 Ruled out

- **#51508 (GDN/KDA recurrent-state corruption)** — **[R]/[D]** requires speculative
  decoding **and** async scheduling to produce the stale zero-accept rows that index
  `-1`. You run neither. Also still open and unmerged.
- **#54649 (Kimi-K3 DSpark/DCP IMA)** — **[R]** requires DSpark spec decoding; the
  mechanism is `MLAAttentionSpec.merge()` relying on an `assert` that optimized-Python
  containers strip. No spec decode, no draft KV-cache spec, no merge.
- **#55924 / the KDA int32 overflow** — **[D]** I diffed `kda.py` between the tags:
  `_store_cache_checkpoints_kernel` is **absent from 0.28.0 entirely**. It cannot be
  your fault. (It becomes relevant only if you upgrade *and* re-enable prefix caching.)
- **#50147** — **[D]** closed by #50386, merged 2026-07-30, which is in your 0.28.0.
  Retained above as a precedent and as the source of the coredump procedure, not as a
  live candidate.
- **#50729, #53000, #52998** — **[D]** all merged before the 0.28.0 cut. Already yours.
- **#53377 (MNNVL allreduce fabric gating)** — as you found, multi-node InfiniBand only.
- **#52225 (Xid 13 warp errors)** — **[R]** SM120 + Nemotron, different hardware and
  model family. Its *isolation advice* is reused above (items 6 and 7); its diagnosis
  is not transferable.
- **#51986 (mnnvl allreduce workspace hang/leak)** — **[R]** IB-only multi-node. Single
  node, not applicable.
- **`TOKENSPEED_MLA` as the culprit** — **[D]** #55780 lists `TokenspeedMLAImpl` among
  the eleven implementations that explicitly declare DCP support, and #51313 shows that
  `TOKENSPEED_MLA` is one of only three backends that can satisfy K3's fp8-KV prefill
  assertion on Blackwell at all. It is the recipe's Blackwell default for both decode
  and prefill. There is no alternative decode backend for this configuration that
  avoids it, and no evidence against it.
- **fp8 KV cache as such** — **[D]** #44044 enabled DCP + fp8 KV in the MLA decode path
  with GSM8K parity; I found no issue claiming a general DCP/fp8-KV incompatibility.
  `--kv-cache-dtype fp8_ds_mla` remains an untested alternative (E17) that takes a bf16
  prefill query and would let you drop `use_prefill_query_quantization`; I am not
  ranking it because I found no evidence it is safer, only that it is different.
- **Very long sequences specifically** — I looked for and did **not** find evidence that
  the fault is tied to approaching the 131072 context limit. The available evidence
  points the other way: #50147 states time-to-crash scales with **concurrency**, and
  your own crashes came at 20-50 min under both 256- and 128-way concurrency. The
  DCP-specific length hazard I did find is the opposite end — **empty shards on short
  sequences** (E19), where a rank with no local KV produces undefined output and LSE.
  Treat "long context" as unproven either way **[I]**.

### 10.7 Safest known-good configuration

If you want the highest probability of a clean 1000x8 run and will pay for it, run the
recipe's Blackwell baseline with DCP and prefix caching removed and every multicast
collective turned off:

```text
--tensor-parallel-size 8
--attention-backend TOKENSPEED_MLA
--attention_config.mla_prefill_backend=TOKENSPEED_MLA
--attention_config.use_prefill_query_quantization=true
--kv-cache-dtype fp8
--no-enable-flashinfer-autotune
--gpu-memory-utilization 0.95
--max-model-len 131072
--max-num-seqs 128
--reasoning-parser kimi_k3
--load-format fastsafetensors
--trust-remote-code
# dropped: --decode-context-parallel-size 8, --dcp-comm-backend a2a,
#          --enable-prefix-caching, --prefix-match-unit 128 (a no-op without it)
```

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ALLREDUCE_USE_FLASHINFER=0
VLLM_ALLREDUCE_USE_SYMM_MEM=0
VLLM_ENGINE_READY_TIMEOUT_S=3600
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
CUDA_COREDUMP_FILE=/weka/.../cuda_coredump_%h.%p.%t
```

This is the recipe's own validated Blackwell profile minus two opt-in features, so it
is the configuration with the most third-party mileage on it **[D]**. Everything
removed is either an opt-in performance feature (DCP) or a no-op for your workload
(prefix caching at a 0.0% hit rate).

**A cheaper first shot**, if you would rather not give up DCP yet: keep your current
serve command exactly as it is and add only the three `VLLM_USE_DIRECT_DCP_*=0` vars
plus `VLLM_ALLREDUCE_USE_SYMM_MEM=0` and the coredump vars. That is items 0, 1 and 3,
costs a few percent, and discriminates the multicast hypothesis from everything else in
one run.

### 10.8 Is it unfixable in 0.28.0?

**Not established either way, and I will not claim otherwise.** What I can say plainly:

- I found **no named, confirmed defect in 0.28.0 that matches your three crashes**.
  There is no issue for Kimi-K3 + DCP8 + `a2a` + `TOKENSPEED_MLA` on sm_103, and no
  issue at all mentioning B300/sm_103 with DCP.
- I did find **one genuine 0.28.0 defect that 0.29.0 fixes** in the exact code path you
  run: the missing capacity fallback in `_init_combine` (E3). That plausibly explains
  Attempt B, and nothing else.
- The prime suspect — the direct symmetric-memory DCP kernels — is **unchanged in
  0.29.0**, so "upgrade" is not the answer to it. It is, however, **fully disableable
  in 0.28.0 by environment variable**, which is why it is item 1.
- If items 1, 2 and 3 all fail, then the remaining honest position is that this is an
  **unreported bug**, and the next step is not another configuration permutation but
  the coredump from item 0, filed as a new issue against `vllm-project/vllm` with the
  faulting kernel named. That is exactly the arc #50147 followed on this same hardware:
  a week of inconclusive tracebacks, then one coredump, then a fix in four days.
