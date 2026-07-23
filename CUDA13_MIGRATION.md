# CUDA 13 / B300 (ai2/holmes) migration notes

Working notes for running open-instruct terminal RL on **CUDA 13 / NVIDIA B300**
(the `ai2/holmes` cluster), migrated from the CUDA 12 / Hopper (`ai2/jupiter`) stack.
Branch: `omni_agent_cuda13` (a worktree off `omni_agent`). Dependency changes were
ported (deps only) from upstream PR #1758 "Upgrade to CUDA 13.0 for B300 support".

TL;DR of what actually bit us, in order: shared weka caches → no-ssh git → cu12-pinned
lock → missing B300 GPU spec → unregistered holmes cluster → missing oe-eval-internal
→ **flash-attn-4 vs Ulysses sequence-parallel on Blackwell**. None were in the cu13
math stack itself — that worked first try.

---

## 0. Reproduce (cu13 holmes launch)

```bash
source env.cuda13.sh   # isolated caches + no-ssh git config (see §1, §2)

# Full build → publish → launch (rebuilds only if image for HEAD commit is absent):
./scripts/train/build_image_and_launch_dirty.sh --cuda-version 13 \
    scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh

# Config-only relaunch (no rebuild) — reuse the already-pushed image:
bash scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh \
    shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13
```

Local (non-Beaker) single-box run, 2 GPUs: `local_rl_2gpu_cuda13.sh` (after sourcing env).

---

## 1. Isolated worktree + caches (don't collide with the cu12 box)

The repo lives on weka **and** `~/.cache` is symlinked to weka, so both are **shared
across machines**. A CUDA-12 session on another box and this CUDA-13 box would otherwise
fight over the same uv env + compiled caches (different CUDA → mutual clobbering).

- Worktree: `/weka/.../open-instruct-cuda13`, branch `omni_agent_cuda13`.
- `env.cuda13.sh` sets isolated `UV_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` (all `*-cuda13`).
- uv `preview = true` stores the env under `$UV_CACHE_DIR/environments-v2/...` (not a
  local `.venv`), so isolating the cache dir **is** the env isolation.
- Leave `~/.cache/huggingface` shared (model downloads, read-mostly). Don't touch
  `.conda` / `.config` (other projects / config-only).

## 2. no-ssh git (container has no `ssh`)

`/root/.gitconfig` rewrites `https://github.com` → `git@github.com:` (and hf.co) via
`insteadOf`, but the container has **no `ssh` binary** → `uv` git-dep clones fail with
`cannot run ssh`.

- Fix: `gitconfig.nossh` (keeps identity + lfs, drops the ssh rewrite), pointed to by
  `GIT_CONFIG_GLOBAL` in `env.cuda13.sh`. Public deps (OLMo-core) then clone over https.
- **Caveat:** this has no credentials, so **private** repos can't be cloned (see §7).

## 3. cu13 dependency stack (pyproject / uv.lock)

Ported from #1758 (deps only, nothing else from `main`):

- Conflicting uv dependency-groups `cuda12` / `cuda13`, `default-groups = ["dev","cuda12"]`.
- Per-group source pins for `torch` / `torchvision` / `torchaudio` / `vllm` / `flash-attn`
  (cu128 index vs cu130 index); `torchvision`/`torchaudio` added as **direct** deps so the
  group pins apply; `flash-attn-3` moved from an inline dep into the per-group blocks
  (cu128 vs cu130 wheel); new `vllm-cu130` index at `wheels.vllm.ai/0.19.1/cu130` (PyPI
  vllm is cu128-only).
- `uv.lock` carries **both** cu128 and cu130 variants, so the cu12 box is unaffected
  (it uses the default `cuda12` group).
- **Always select the group explicitly on this box:** `uv sync --no-default-groups
  --group dev --group cuda13`. A bare `uv sync` uses the default `cuda12` group and fails
  building cu128 `causal-conv1d` against the system nvcc 13.

## 4. Dockerfile

- Multi-stage `ARG CUDA_VERSION` (12/13) selector. Kept **ubuntu24.04** (our DOCA/OFED +
  podman-from-source stack targets `ubuntu2404`; upstream #1758 was on 22.04) →
  `nvidia/cuda:13.0.3-devel-ubuntu24.04`.
- The nltk/deps step runs `uv ... --no-default-groups --group dev --group cuda${CUDA_VERSION}`.

## 5. `build_image_and_launch_dirty.sh --cuda-version 12|13`

Mirrors #1758's flag plumbing:
- Passes `--build-arg CUDA_VERSION`, suffixes the image name `-cuda${v}`
  (→ `<user>/open-instruct-integration-test-<branch>-cuda13`), and pins the **local**
  `uv sync` to the matching group (so launching from this cu13 box doesn't fail).
- **Launch is gated on the push:** with `set -euo pipefail`, `beaker image create` (the
  push) must succeed before the mason submit runs.
- **Force a rebuild** when the working tree changed but the commit didn't (e.g. after
  copying in `oe-eval-internal`): delete the Beaker image
  (`beaker image delete <user>/<image>`). The dirty script skips the build when an image
  whose description contains the current commit hash already exists.
- **`jq` is required** (Beaker JSON parsing) and was missing on the dev box —
  `apt-get install -y jq`.

## 6. Required-to-run code fixes (values from #1758)

Hardware/cluster enablement — the code hard-fails without these:
- **B300 in `GPU_SPECS`** (`utils.py`): `"b300": {flops 2250e12, mem 288e9, bw 8e12}`.
  Without it grpo_fast raises `Unknown device name: NVIDIA B300 SXM6 AC` at startup
  (`NVIDIA B300 SXM6 AC` normalizes to `b300` by substring). + `test_utils.py` cases.
- **`ai2/holmes`** added to `launch_utils.WEKA_CLUSTERS` **and** `INTERCONNECT_CLUSTERS`,
  else `mason.py` rejects `--cluster ai2/holmes` / won't mount weka / blocks multi-node.

## 7. `oe-eval-internal` (fresh-worktree gap)

`grpo_fast.py` calls `utils.check_oe_eval_internal()` which raises **unconditionally when
running in Beaker** (gated only on `BEAKER_EXPERIMENT_ID`; `--try_launch_beaker_eval_jobs_on_weka
False` does **not** skip it). A fresh worktree lacks the dir; the Dockerfile's optional
`COPY oe-eval-interna[l]` silently copies nothing.

- It's a **private** repo → can't `git clone` here (no ssh + no creds, §2). Fix by
  **copying from a sibling checkout**: `cp -r <other>/oe-eval-internal ./ && rm -rf
  oe-eval-internal/.git` (37 MB without .git). Then force a rebuild (§5).
- Not gitignored, but copying it doesn't change the commit hash.

## 8. Attention backend on Blackwell + sequence parallelism — the key one

**What each stack ends up using (attn auto-detect, `model_utils.detect_attn_implementation`):**
picks `flash_attention_4` iff fa4 wheel present **and GPU compute major ≥ 10** (Blackwell);
else fa3 if `major ≥ 9`; else fa2; else sdpa.
- **cu12 production (Hopper, major 9) → `flash_attention_3`** (prod scripts set no
  `--attn_implementation`). fa3 *is* the "hopper" impl, so its kernels match Hopper → works.
- **B300 (major 10) → `flash_attention_4`** (confirmed: local log `Auto-detected ... flash_4`).

On B300 with **Ulysses SP** (`--sequence_parallel_size > 1`) you hit a Blackwell catch-22,
found by trial across three holmes launches:

1. **fa4 → SP rejects it.** `deepspeed/.../ulysses_sp.py` has a hardcoded allowlist
   `supported_attn_implementation = ["flash_attention_2","flash_attention_3","sdpa"]` and
   raises `ValueError: flash_attention_4 ... isn't currently supported by Ulysses sequence
   parallelism`. Ulysses wraps a registered HF attention *function* (`ALL_ATTENTION_FUNCTIONS`)
   with all-to-all resharding; fa4's new `flash_attn.cute` interface isn't adapted/allowlisted.
   It's a **deepspeed software gap, not fundamental** — fa4 itself runs fine on B300 (the
   local **SP=1** run used fa4 and passed). So: **no SP → fa4 is great; SP → fa4 blocked.**
2. **fa3 → no Blackwell kernel.** fa3 passes the SP allowlist but its "hopper" kernels aren't
   built for sm_103: `CUDA error (flash-attention/hopper/flash_fwd_launch_template.h): no
   kernel image is available for execution on the device`, in the learner's ZeRO-3 optimizer
   priming. (This is why prod's implicit fa3 can't just be reused on B300.)
3. **fa2 → works.** SP-compatible **and** the `flash_attn 2.8.3+cu130` wheel has Blackwell
   kernels (verified locally: `flash_attn_func` runs on the B300). **This is the fix.**

- **Fix:** `--attn_implementation flash_2` on B300 + SP. Config-only → reuse the image.
- **Perf note:** on B300 you're capped at fa2 whenever SP is on (fa4 > fa3 > fa2 in speed).
- **Path to fa4+SP (deepspeed bump — verified upstream):** we're on **deepspeed 0.18.4**,
  whose `ulysses_sp.py` uses the hardcoded allowlist. Deepspeed **master already refactored**
  this to a blocklist `unsupported_attn_implementation = ["eager","paged|eager"]` and otherwise
  validates against `transformers.ALL_ATTENTION_FUNCTIONS` — and `flash_attention_4` IS in that
  registry (transformers 5.4.0). So a newer deepspeed would **permit fa4 with Ulysses SP**.
  Follow-up: bump deepspeed (re-lock + rebuild), drop the `--attn_implementation flash_2`
  override (auto-detect → fa4), and verify fa4 runs correctly through Ulysses' all-to-all at
  runtime (permitted ≠ validated). Confirm which released version has the refactor (0.18.4 lacks it).
- **Candidate codebase fix (TODO):** make `detect_attn_implementation()` SP-aware — when
  sequence parallelism is enabled, don't pick fa4 (and don't pick fa3 on Blackwell where it
  has no kernel) → default to fa2, so SP runs on B300 work without the override.
- **`main` is affected too (upstream candidate).** `origin/main` pins the same
  `deepspeed>=0.18.3` (→0.18.4, old allowlist) and the same `flash-attn-4` wheel + fa4
  auto-detect (all introduced by #1758). So SP training on B300 from `main` hits the identical
  crash — this is not fork-specific. #1758's B300 GPU tests didn't exercise SP>1, so it went
  unnoticed. The detect fix and/or deepspeed bump belong upstream.

## 9. Registry mirror (verify before every sandbox-RL launch)

Sandbox images are pulled through a docker.io pull-through cache set by
`--env MIRROR_URL`. A dead mirror hides as "working" (silent docker.io fallback).

- **LIVE:** `jupiter-cs-aus-112.reviz.ai2.in:5000` (v2=200, pull-through OK; reachable
  even from the holmes dev box).
- **DEAD now:** `jupiter-cs-aus-137` (conn-fail) — the cu12 `qwen35_9b_dppo_repro.sh`
  hardcodes `-137` and calls it "live"; that comment is **stale**, update to `-112`.
  `-193` also dead (long known).
- Check: `curl -m8 http://<host>/v2/` (want 200) and a `.../manifests/3.12-slim` pull-through.

## 10. Validation status

| Run | Config | Result |
|-----|--------|--------|
| Local 2-GPU | SP=1, stage 2, `swerl_sandbox`, Qwen3-0.6B, cu130/B300 | **PASSED** — 2 steps, exit 0, wandb `lvb9a3to` |
| holmes `01KY6KQZ…` | 4-GPU vanillux+DPPO+SP=2 | crashed: missing `oe-eval-internal` (§7) |
| holmes `01KY6MFA…` | ” (image w/ oe-eval-internal) | crashed later: fa4 vs SP (§8) |
| holmes `01KY6N9F…` | ” + `--attn_implementation flash_3` | launched, in progress |

The cu13 image, holmes scheduling, Ray, and the cu130 vLLM engine (flashinfer, CUDA
graphs, KV cache) all came up fine — every failure was a config/hardware-enablement gap,
not the CUDA-13 math stack.
