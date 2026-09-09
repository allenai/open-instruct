# CUDA 13 / B300 (ai2/holmes) migration notes

Working notes for running open-instruct terminal RL on **CUDA 13 / NVIDIA B300**
(the `ai2/holmes` cluster), migrated from the CUDA 12 / Hopper (`ai2/jupiter`) stack.
Branch: `omni_agent_cuda13` (a worktree off `omni_agent`). Dependency changes were
ported (deps only) from upstream PR #1758 "Upgrade to CUDA 13.0 for B300 support".

TL;DR of what actually bit us, in order: shared weka caches → no-ssh git → cu12-pinned
lock → missing B300 GPU spec → unregistered holmes cluster → missing oe-eval-internal
→ **flash-attn-4 vs Ulysses sequence-parallel on Blackwell**. None were in the cu13
math stack itself — that worked first try.

**fa4 + Ulysses SP on B300 is now RESOLVED (2026-07-27)** via two dependency bumps —
**deepspeed 0.18.4→0.19.3** (permits fa4 in Ulysses) and **flash-attn-4 b5→b23** (adds the
head_dim=256 Blackwell kernel Qwen3.5 needs). Validated end-to-end on the 9B; see §8 + §10.
The `--attn_implementation flash_2` workaround is no longer required (kept as a fallback).

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

- **Interim fix (what the successful fa2 runs used):** `--attn_implementation flash_2` on B300 + SP.
  Config-only → reuse the image. fa2's `flash_attn 2.8.3+cu130` wheel has Blackwell kernels for
  every head dim we use (incl. Qwen3.5's hd256). **Now superseded by the fa4 path below**, but
  remains a valid fallback.

### fa4 + Ulysses SP on B300 — RESOLVED (2026-07-27), a TWO-part fix

fa4 with SP on B300 was blocked by **two independent gaps**. Both are now fixed; bump both deps:

1. **deepspeed allowlist → permit fa4 in Ulysses.** deepspeed **0.18.4**'s `ulysses_sp.py` used the
   hardcoded allowlist that rejected fa4 (point 1 above). **deepspeed ≥0.19.0** (PR #7887, first in
   v0.19.0 2026-05-06; we pin **0.19.3**) replaced it with a *blocklist*
   `unsupported_attn_implementation = ["eager","paged|eager"]` and otherwise validates against
   `transformers.ALL_ATTENTION_FUNCTIONS` (which registers `flash_attention_4`). ⇒ Ulysses now
   **permits** fa4. Verified: the old `ValueError` is gone and fa4 reaches its actual kernel.
2. **flash-attn-4 hd256 Blackwell kernel → the second, deeper gap.** Once *permitted*, fa4 on
   **Qwen3.5-9B (head_dim=256)** still crashed *inside* the fa4 kernel at the learner's ZeRO-3 dummy
   step: `AssertionError: (head_dim, head_dim_v)=(256, 256) is not supported on SM100/SM110`. fa4's
   `flash_attn.cute` (which OWNS `flash_attn/cute/`, provided by the **flash-attn-4** package —
   *not* the `flash-attn` 2.8.3 wheel) had **no head_dim>128 Blackwell kernel** in our pinned
   **4.0.0b5**. Upstream added a dedicated **hd256 SM100 kernel** on 2026-04-23 (PR #2412,
   `sm100_hd256_2cta_fmha_*`), shipped in later betas. **Bump the pin b5 → b23**
   (`fa4-v4.0.0.beta23`). flash-attn-4 is pure-Python CuTeDSL that **JIT-compiles kernels at
   runtime** → no cu130 wheel to find, no source build; a version bump is the whole fix. (Qwen3-0.6B
   at hd128 was never affected.) Caveat: fa4 is still **beta**; hd256 fwd perf is being tuned upstream
   (#2576), so benchmark vs fa2 before assuming it's faster.

- **Validation (2026-07-27, see §10):** *(a)* local **standalone** fa4 hd256 fwd+bwd match SDPA to
  rel ~3e-3 on the B300; *(b)* on holmes the 9B SP=2 smoke (deepspeed 0.19.3 + flash-attn-4 b23)
  passed the **ZeRO-3 dummy step — a full fwd+bwd through fa4's hd256 kernel *inside* Ulysses'
  all-to-all** — then weight-sync + into the training loop with **no crash**. So fa4+SP+hd256 is
  proven end-to-end. (A nonzero reward-weighted step is still gated on rollout variance, §11 —
  orthogonal to fa4.)
- **Backend choice on B300+SP now:** with the two deps above, **fa4 (fastest) works for all head
  dims** — `--attn_implementation flash_4` (or drop the override; auto-detect picks fa4). Keep
  `flash_2` only as a fallback. **fa3 still has no Blackwell kernel — never use it on B300.**
- **Candidate codebase fix (still nice-to-have):** make `detect_attn_implementation()` refuse fa3
  on Blackwell (no kernel). fa4 auto-detect is now correct for SP given the deps.
- **`main` is affected too (upstream candidate).** `origin/main` pins `deepspeed>=0.18.3` (old
  allowlist) **and** flash-attn-4 b5 (no hd256 Blackwell kernel), both via #1758 — so SP+hd256
  training on B300 from `main` hits *both* crashes in turn. #1758's B300 GPU tests didn't exercise
  SP>1 or hd256. Both bumps (deepspeed ≥0.19.0, flash-attn-4 ≥b23) belong upstream.

## 9. Registry mirror (verify before every sandbox-RL launch)

Sandbox images are pulled through a docker.io pull-through cache set by
`--env MIRROR_URL`. A dead mirror hides as "working" (silent docker.io fallback).

- **LIVE (2026-07-27):** `jupiter-cs-aus-102.reviz.ai2.in:5000` (v2=200, `python:3.12-slim`
  pull-through=200). All cu13 scripts now point here.
- **DEAD:** `-112` (was live 2026-07-23, dead by 2026-07-27), `-137`, `-193`. **Mirrors move —
  re-check every launch.** The cu12 `qwen35_9b_dppo_repro.sh` hardcodes `-137` and calls it
  "live"; that comment is stale.
- Check: `curl -m8 http://<host>/v2/` (want 200) and a `.../manifests/3.12-slim` pull-through.

## 10. Validation status (2026-07-23; fa4 update 2026-07-27)

| Run | Config | Result |
|-----|--------|--------|
| Local 2-GPU | SP=1, stage 2, `swerl_sandbox`, Qwen3-0.6B | **PASSED** — 2 steps, exit 0, wandb `lvb9a3to` |
| holmes `01KY6KQZ` | 4-GPU vanillux+DPPO+SP=2, Qwen3-0.6B | crashed: missing `oe-eval-internal` (§7) |
| holmes `01KY6MFA` | ” (image w/ oe-eval-internal) | crashed later: fa4 vs SP allowlist (§8) |
| holmes `01KY6N9F` | ” + `--attn_implementation flash_3` | crashed later: fa3 no B300 kernel (§8) |
| holmes `01KY6VRG/VSP` | 4-GPU + `flash_2`, Qwen3-0.6B (4k) | **infra PASSED** exit 0; empty batch (§11) |
| holmes 9B smokes | 4-GPU `flash_2`, tmax **9B `_cg`** (`step_360_cg`), 4k & 64k | **infra PASSED** exit 0; empty batch (§11) |
| holmes `01KY6Z9F` | **full prod 4-node/32-B300**, `flash_2`, 64k, SP=4, DPPO, `hamishivi/Qwen3.5-9B`, 8×32 rollouts, `total_episodes 6400` | **PASSED — real gradient steps**: `scores 0.83`, `advantages ∈ [-0.94, 0.38]`, non-empty batches, `seq_len_max 65536` |
| **fa4 local scratch** | standalone fa4 **hd256** fwd+bwd on B300, flash-attn-4 **b23** | **PASSED** — matches SDPA rel ~3e-3, no assert |
| **holmes `01KYK2ZF`** | 4-GPU SP=2, 9B `step_360_cg`, `flash_4`, deepspeed 0.19.3 + flash-attn-4 **b5** | crashed: fa4 hd256 assert on SM100 (b5 lacks kernel, §8) |
| **holmes `01KYK6G0` (AS-on) / `01KYK7WM` (noAS)** | ” + flash-attn-4 **b23** | **fa4+SP+hd256 PASSED** — ZeRO-3 dummy step (fwd+bwd through Ulysses) + weight-sync + training loop, exit 0, no crash; empty batch (§11) |

Every failure along the way was a config / hardware-enablement / dependency-version gap — the
CUDA-13 math stack (torch cu130, vLLM cu130 flashinfer/CUDA-graphs/KV, deepspeed) came up first
try each time. The fa4-on-B300 path required **two** dependency bumps (deepspeed ≥0.19.0 to permit
fa4 in Ulysses, flash-attn-4 ≥b23 for the hd256 Blackwell kernel); see §8.

## 11. Reward variance ≠ context length (why smokes had empty batches)

The 0.6B and small-N 9B smokes ran the full pipeline to `exit 0` (model load → weight sync
→ rollouts → training loop → model save) but **skipped the gradient step**:
```
All prompts were filtered during accumulation. Filtered: N (zero std: N, solved: 0, nonzero: 0)
🤡 After packing, there is not enough data to train → Empty batch, skipping training step
```
GRPO/DPPO drops any prompt-group with **zero within-group reward std** (needs a solved/unsolved
mix to form advantages). With only `num_unique_prompts_rollout=4` and all rollouts scoring 0,
every group had zero variance → empty batch. **This was NOT context length** — even the 64k
smoke hit it (0 solves in 16 rollouts). The fix is enough prompts × samples to get variance:
the **full prod run (8×32 = 256 rollouts/step) landed real steps immediately** (`scores 0.83`,
advantages spanning −0.94…0.38). So for a smoke that must show a real weight update, use the
production rollout width, not a tiny one.
