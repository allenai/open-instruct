# Trainer warm-up fixes (FLA per-pack-length autotune), 2026-11-18

Investigation and prototype only: nothing was launched, the running runs were not touched, and the open-instruct and miles-olmo-core trees are clean.

## Recommendation

Use **FLA's own config cache** with `FLA_CACHE_MODE=default` and a pre-built config directory. It needs only two `[launch.env]` lines: no code change and no image rebuild. It applies to every future run and every auto-resume.

Validate it first with one 1-GPU benchmark job (spec below). Fix (c), the compiler-cache patch, is optional and complementary but needs an image rebuild. Padding (a) and a startup pre-tune (b) are not worth it.

## What causes the warm-up (confirmed in the sources)

- **Sources read:**
  - fla-core 0.5.2 wheel, in the scratchpad.
  - miles-olmo-core branch `codex/olmicro-mainline-sync` @ `b83a66f`, the image's MILES.
- **Packing.** `miles/backends/core_utils/packing.py` builds unpadded, order-preserving packs:
  - `plan()` packs greedily up to `core.packing_max_tokens` (16,384 / 24,576 / 40,960 for 8k / 16k / 32k).
  - `combine()` concatenates the samples into a `[1, T]` pack with `doc_lens`. So B = 1 and T = the pack's actual token count, which differs on almost every pack.
  - Observed fill fraction is 0.68–0.99, typically about 0.8.
- **Kernels whose autotune key depends on pack length** (Olmicro GDN path; NB is also a `tl.constexpr`, so every new NB is a new kernel specialization):

  | Kernel | Decorator | Key | NB formula | Distinct keys |
  |---|---|---|---|---|
  | `causal_conv1d_fwd_kernel`, `causal_conv1d_bwd_kernel` | `fla_cache_autotune` | `['D','W','NB']` | `cdiv(B*T, 1024)` (`fla/modules/conv/triton/ops.py`) | up to 40 at 32k, for each of D = 2048 and 4096 |
  | `l2norm_fwd_kernel`, `l2norm_bwd_kernel` | `fla_cache_autotune` | `['D','NB']` | `cdiv(rows, 65536)` | ~10 |
  | `layer_norm_gated_fwd/bwd_kernel` (`fused_norm_gate.py`) | plain `triton.autotune` | `[..., 'NB', ...]` | `cdiv(rows, 65536)` | ~10 |

  The GDN chunk kernels' keys depend only on H, K, V and BT, so they tune once.
- **Measured cost per new key** (`monitor/bench-shape-churn-20261118/flash_4.log`):
  - conv fwd ≈ 6.3 s and conv bwd ≈ 9.8 s per D value, about 32 s per new conv NB;
  - l2norm fwd + bwd ≈ 20 s per new NB;
  - layer_norm_gated fwd + bwd ≈ 11 s per new NB.

  Conv always chose `BD=32, num_warps=4`, except one small-NB fwd key.
- **Observed excess training time from warm-up** (training s/update minus tokens at the steady rate): 32k ≈ **89 min**, 8k ≈ 33 min, 16k ≥ 25 min (still converging at the time of measurement).
  - This is larger than the per-rank tuning total (about 27 min at 32k) because each step waits for the slowest of the 4 ranks, and the ranks meet new pack lengths at different times.
  - Each auto-resume pays the cost again, because Triton's per-process results live in node-local `/tmp/core-triton-*`.

## Options

### Option A: FLA config cache (recommended; config/env only)

`fla/ops/utils/cache.py` (`CachedAutotuner`) checks `$FLA_CONFIG_DIR/{kernel}.json` before autotuning for every `fla_cache_autotune` kernel.
- `FLA_CACHE_MODE=default` returns that file's `default_config` for every key (any D, NB or dtype), so the kernel never autotunes.
- Kernels without a file, and plain `triton.autotune` kernels, are unchanged.
- I verified the lookup by running FLA's own `cache.py` against the generated directory: DEFAULT returns the config for NB 1/17/40/63 under both the bf16 and fp32 weight signatures, and returns None for kernels with no file. In STRICT mode an unseen NB falls back to autotune.

**Built:**
- `tools/make_fla_config_cache.py` turns `TRITON_PRINT_AUTOTUNING=1` logs into an FLA config directory. The default is the winner at the largest NB, and all observed keys are kept as exact-match entries.
- `fla-configs/b300-fla0.5.2-v1/` holds files for four kernels:

  | Kernel | Default config |
  |---|---|
  | `causal_conv1d_fwd_kernel` | `BD=32, num_warps=4` |
  | `causal_conv1d_bwd_kernel` | `BD=32, num_warps=4` |
  | `l2norm_fwd_kernel` | `BT=64, num_warps=16` |
  | `l2norm_bwd_kernel` | `BT=8, num_warps=2` |

  All use `num_stages=3, num_ctas=1`.

**Run-config diff** (for future runs; add under `[launch.env]`):

```toml
# Fixed configs for FLA's pack-length-keyed conv/l2norm kernels: no per-pack-length autotune.
FLA_CACHE_MODE = "default"
FLA_CONFIG_DIR = "/weka/oe-adapt-default/jacobm/olmicro/fla-configs/b300-fla0.5.2-v1"
# First run only, to verify no causal_conv1d/l2norm autotune lines remain in the trainer log:
TRITON_PRINT_AUTOTUNING = "1"
```

- These are allowed `launch.env` names (`run_spec.py` only reserves launcher/rank variables), and they go straight into the Beaker task env.
- `tools/make_v2_refresh_config.py` does not add them yet.

**Expected saving:**
- Autotune for conv and l2norm goes to zero. What remains:
  - one JIT compile of the chosen config per new NB per kernel (estimated about 1–2 s, against 6–14 s for a full autotune);
  - layer_norm_gated autotune, about 2 min in total;
  - one-time chunk-kernel tuning, about 2 min.
- Warm-up should drop from about 30–90 min to about 5–10 min per run and per resume. That is an estimate until validated.

**Risks and checks:**
1. **Performance:** one config is used for every D and NB. The benchmark winners support it for conv; the l2norm winners were noisy, but that kernel is tiny. Validation should show warm throughput within about 3% of baseline.
2. **Correctness:** configs only change tiling (masked kernels, values from the kernels' own candidate lists). The benchmark now prints a seeded loss per pack, so parity between arms is checked.
3. **Triton version:** for Triton ≥ 3.5.1, FLA builds `triton.Config` with `num_ctas`, `maxnreg` and `ir_override` (the files include `num_ctas` and `maxnreg`). The validation job prints the image's Triton version.
4. **Env propagation:** Ray trainer actors should inherit the container env. Confirm in the first run with `TRITON_PRINT_AUTOTUNING=1`.
5. **Scope:** any process that imports `fla` (possibly SGLang) also uses these four configs. That is safe but could be slower.
6. **Other model sizes:** smaller sizes have different D values; DEFAULT applies the same generic config. If needed, regenerate per size from a short benchmark.

### Option C: compiler-cache wrapper accepts external Core configs (optional; image rebuild)

- **Why restore is refused:** `miles/utils/compiler_cache/compiler_cache.py:fingerprint()` raises "External Core model-config files are not qualified by this wrapper" whenever `core.model_config` (a path) is set. The path's contents are not part of the fingerprint.
- **Patch:** `tools/patches/miles-compiler-cache-external-core-config.patch` (not applied; against `codex/olmicro-mainline-sync` b83a66f). In `startup_cache.prepare()` it replaces the path with `core.model_config_sha256` (the file's content hash), so the identity covers the architecture and the check passes. It needs maintainer review and an image rebuild.
- **Benefit:** restores compiled kernels and Triton autotune results (`FLA_CACHE_RESULTS=1` by default) through the existing shared root. The fingerprint includes the packing budget, caps and model, so it only helps identical-config reruns and auto-resumes. That makes it complementary to Option A, removing the residual compiles and layer_norm_gated tuning on resume.
- **Rejected env-only alternative:** pointing `TRITON_CACHE_DIR` at Weka. The wrapper overrides `TRITON_CACHE_DIR` per worker and forbids shared mutable caches (`local_environment` refuses /weka). Disabling the wrapper to do it anyway is unqualified: many concurrent small-file writes on Weka.

### Option A′: pad packs to fixed buckets (not recommended)

- It would need a padding document in `packing.combine()` plus masking through the loss path, and an image rebuild.
- At the observed fill of about 0.8, padding to the full budget costs about 20% more trainer compute, and bucketing still autotunes once per bucket.
- Option A removes the autotune without wasting compute.

### Option B: startup pre-tune pass (not recommended)

- Without persistence it only moves the 30–90 min to the start of the run.
- The FLA config directory is effectively a persisted pre-tune.

## Validation plan (single GPU; not submitted, since we are at the 3-node limit)

- **Spec:** `tools/bench_fla_config_cache_beaker.json`: urgent, 1 GPU, `minRuntime` 1h, same hostname allowlist (511 excluded) and image as the earlier churn benchmark.
- **What it runs:** `tools/bench_shape_churn.py` (which now takes `--seed` and logs loss) twice on the same 8 real-length 40,960-token packs, cold then warm:
  1. baseline;
  2. `FLA_CACHE_MODE=default FLA_CONFIG_DIR=.../b300-fla0.5.2-v1`.

  Both arms set `TRITON_PRINT_AUTOTUNING=1`. Output goes to `monitor/bench-fla-config-cache/`.
- **Pass criteria:**
  - no `Triton autotuning for function causal_conv1d_*` or `l2norm_*` lines in the fix arm;
  - cold-pack time in the fix arm ≤ about 5 s (baseline 33–71 s);
  - warm tok/s within 3% of baseline (26–29.5k);
  - loss per pack matches baseline to bf16 tolerance (relative ≤ 1e-3).
- **Submit with:** `beaker experiment create tools/bench_fla_config_cache_beaker.json --workspace ai2/olmo3p5-training`, once a GPU is free.
