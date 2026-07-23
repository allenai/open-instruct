# Playbook: get flash-attn-4 working with Ulysses SP on B300 (deepspeed/transformers bump)

**Goal.** On B300/holmes, sequence-parallel (`--sequence_parallel_size > 1`) training is
currently forced onto **flash-attn-2** (`--attn_implementation flash_2`) because fa4 (the
fastest, Blackwell-native backend, which the code auto-detects on B300) is blocked by the
**deepspeed 0.18.4** Ulysses allowlist, and fa3 has no Blackwell kernel. This playbook bumps
deepspeed (and, if needed, transformers) so **fa4 works with Ulysses SP**, then rebuilds the
cu13 image and re-runs the holmes smoke to validate — restoring the fastest attention on B300.

Root-cause detail + the full attention saga live in **`CUDA13_MIGRATION.md` §8**. Read that
first. This doc is the *how-to-execute* companion (env, edits, build, publish, smoke, monitor).

---

## 0. Why this is fixable (the key facts, verified 2026-07-23)

- `model_utils.detect_attn_implementation()` picks `flash_attention_4` on GPU compute
  major ≥ 10 (B300 = sm_103). fa4 itself **runs on B300** (the local SP=1 smoke used it, passed).
- deepspeed **0.18.4** `runtime/sequence_parallel/ulysses_sp.py` hard-rejects fa4 via
  `supported_attn_implementation = ["flash_attention_2","flash_attention_3","sdpa"]`.
- deepspeed **master** already refactored this to a *blocklist*
  `unsupported_attn_implementation = ["eager","paged|eager"]` and otherwise validates against
  `transformers.ALL_ATTENTION_FUNCTIONS`.
- transformers **5.4.0** already registers `flash_attention_4` in `ALL_ATTENTION_FUNCTIONS`.
- ⇒ A deepspeed version with the blocklist refactor should **permit** fa4+SP. **Permitted ≠
  validated** — must confirm fa4's interface actually runs through Ulysses' all-to-all at runtime.
- This affects **`origin/main`** too (same `deepspeed>=0.18.3` pin + same fa4 wheel/auto-detect
  from #1758); #1758's B300 GPU tests didn't exercise SP>1. So a working fix is an upstream PR.

---

## 1. Environment (pick up cold)

- **Box:** `holmes-cs-aus-*` — NVIDIA **B300** (sm_103, driver 590, CUDA 13.0 toolkit), 2 GPUs
  on the dev box. Holmes compute nodes have 8.
- **Worktree:** `/weka/nora-default/shashankg/code/open-instruct-cuda13`, branch
  `omni_agent_cuda13`. (Separate from the cu12 session's `open-instruct/` checkout — both on
  shared weka.)
- **ALWAYS first:** `cd` to the worktree and `source env.cuda13.sh`. It sets isolated
  `UV_CACHE_DIR=~/.cache/uv-cuda13` (+ `TRITON_/TORCHINDUCTOR_/TORCH_EXTENSIONS_*-cuda13`) and
  `GIT_CONFIG_GLOBAL=<worktree>/gitconfig.nossh`. Rationale: weka `~/.cache` + repo are shared
  with the cu12 box → without isolation the uv env / compiled caches clobber each other; and
  the container has **no `ssh`** while `/root/.gitconfig` rewrites https→ssh (so public git deps
  need `gitconfig.nossh` to clone over https). uv `preview=true` stores the env under
  `$UV_CACHE_DIR/environments-v2/...` (not `.venv`).
- Prereqs already handled but re-check on a fresh box: `jq` (`apt-get install -y jq`), docker
  rootless (data root `/media/16TBNVME`), `oe-eval-internal/` present in the worktree (see §5).
- **cu13 env build:** `uv sync --no-default-groups --group dev --group cuda13`. A bare
  `uv sync` uses the default `cuda12` group and fails (builds cu128 causal-conv1d vs nvcc13).

---

## 2. Find the deepspeed version to bump to

The blocklist refactor is in deepspeed master; find the earliest **released** tag that has it,
or pin master.

```bash
# Which released version has the blocklist? Inspect the file across tags on GitHub:
#   https://github.com/deepspeedai/DeepSpeed/commits/master/deepspeed/runtime/sequence_parallel/ulysses_sp.py
# Look for the commit that replaces `supported_attn_implementation` with
# `unsupported_attn_implementation = ["eager","paged|eager"]`, then the first release tag after it.
# (WebFetch the raw file at a tag to confirm, e.g.
#  https://raw.githubusercontent.com/deepspeedai/DeepSpeed/v0.19.X/deepspeed/runtime/sequence_parallel/ulysses_sp.py )
```

Two ways to pin in `pyproject.toml`:
- **Released tag (preferred):** bump `"deepspeed>=0.18.3"` → `"deepspeed>=<X>"`.
- **Git rev (if unreleased):** add to `[tool.uv.sources]`:
  `deepspeed = { git = "https://github.com/deepspeedai/DeepSpeed.git", rev = "<sha>" }`.

Note vllm 0.19.1 also depends on deepspeed indirectly? (No — but vllm pins other libs.) Watch
the `uv lock` resolution for conflicts, and for deepspeed API changes that touch
`grpo_fast.py` / `olmo_core_*` (deepspeed is a core dep).

---

## 3. Edit + re-lock + local sync

```bash
source env.cuda13.sh
# edit pyproject.toml deepspeed pin (and [tool.uv.sources] if git-pinning)
uv lock                                                   # re-resolve BOTH cuda12 + cuda13 groups
uv sync --no-default-groups --group dev --group cuda13    # rebuild the cu13 env
```

Verify the new deepspeed + that fa4 is no longer allowlisted-out:
```bash
uv run --no-sync python -c "import deepspeed; print(deepspeed.__version__)"
DS=$(uv run --no-sync python -c "import deepspeed,os;print(os.path.join(os.path.dirname(deepspeed.__file__),'runtime/sequence_parallel/ulysses_sp.py'))")
grep -nE "unsupported_attn_implementation|supported_attn_implementation|flash_attention_4" "$DS"
uv run --no-sync python -c "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as A; print('fa4' , 'flash_attention_4' in A.valid_keys())"
```

## 4. Local sanity

- fa4 on B300 already proven (local SP=1 smoke). The thing that MUST be validated is **fa4
  through Ulysses SP at runtime** — and SP=2 needs 4 GPUs (2 learners SP=2 + 2 engines); the
  dev box only has 2, so a faithful SP test isn't possible locally → **validate on holmes (§7).**
- Cheap local check before spending a holmes slot: import deepspeed's updated `register_with_transformers`
  path / confirm no allowlist rejection for `flash_attention_4` (grep in §3). Optionally run the
  repo unit tests that touch attention/model load: `uv run --no-default-groups --group dev --group cuda13 pytest open_instruct/test_utils.py -k device -q`.

## 5. Build + publish the cu13 image (deepspeed bump ⇒ REBUILD required)

A dep/lock change is **not** config-only — the image must rebuild (the `uv sync` layer re-runs;
base/apt/podman-from-source stay cached).

```bash
# oe-eval-internal must exist in the worktree (private repo; can't clone here — copy from a sibling):
[ -d oe-eval-internal ] || { cp -r /weka/nora-default/shashankg/code/open-instruct/oe-eval-internal ./ ; rm -rf oe-eval-internal/.git ; }

# Commit first (user OK's commits, NO push). New commit → new hash → dirty launcher rebuilds:
git add pyproject.toml uv.lock && git commit -m "cuda13: bump deepspeed to <X> for fa4+Ulysses-SP on B300"

# Build (cu130 base + --group cuda13) → push as ...-cuda13 → mason submit, one shot:
source env.cuda13.sh
./scripts/train/build_image_and_launch_dirty.sh --cuda-version 13 \
    scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh
```

- `--cuda-version 13` passes `--build-arg CUDA_VERSION=13` and names the image
  `<user>/open-instruct-integration-test-omni_agent_cuda13-cuda13`.
- **Launch is gated on the push** (`set -e`): mason only runs after `beaker image create`.
- If you rebuilt at the *same* commit and the launcher skips the build, force it:
  `beaker image delete <user>/open-instruct-integration-test-omni_agent_cuda13-cuda13`.
- `jq` and Beaker auth (`beaker account whoami`) must work locally.

## 6. Flip the smoke script to fa4

In `scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh`, to test fa4:
- **Remove** `--attn_implementation flash_2` (let auto-detect → fa4 on B300), **or**
- set `--attn_implementation flash_4` explicitly.

Everything else stays: 4 GPUs / 1 node = 2 learners (SP=2, stage 3) + 2 vLLM engines,
`swerl_vanillux_sandbox` + `allenai/tmax-15k-open-instruct` + DPPO (tv 0.1) + liger +
lm_head_fp32, Qwen3-0.6B, small ctx, `MIRROR_URL=jupiter-cs-aus-112`.

**Mirror pre-check (do every launch — mirrors move):**
```bash
host=jupiter-cs-aus-112.reviz.ai2.in:5000
curl -sS -m8  -o /dev/null -w "v2 %{http_code}\n" "http://$host/v2/"                          # want 200
curl -sS -m20 -o /dev/null -w "manifest %{http_code}\n" \
  -H "Accept: application/vnd.oci.image.index.v1+json" "http://$host/v2/library/python/manifests/3.12-slim"  # 200
```
If the image already exists for the commit and you only changed the launch script (config-only),
skip the rebuild — launch directly:
```bash
source env.cuda13.sh
bash scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh \
    shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13
```

## 7. Monitor + diagnose the holmes run

Grab the `beaker.org/ex/<EXP>` URL from the launch output. Arm a state monitor (poll the API;
do **not** grep rollout logs for errors — terminal-RL logs are full of benign "Killed Docker
container"/"timed out" noise):

```bash
EXP=<id>; prev=""
while true; do
  j=$(beaker experiment get "$EXP" --format json 2>/dev/null) || { sleep 30; continue; }
  st=$(echo "$j" | jq -rc '.[0].jobs[-1].status // {}')
  phase=$(echo "$st" | jq -r 'if .finalized then "finalized" elif .started then "running" elif .scheduled then "scheduled" else "created" end')
  code=$(echo "$st" | jq -r '.exitCode // empty'); n=$(echo "$j" | jq -r '.[0].jobs|length')
  line="phase=$phase exit=${code:-none} jobs=$n"; [ "$line" != "$prev" ] && { echo "$(date +%H:%M:%S) $line"; prev="$line"; }
  [ "$phase" = finalized ] && { echo "DONE exit=${code:-?} jobs=$n"; break; }; sleep 30
done
```

**Read the signal by timing** (from the smoke saga): a job that flips `running`→`scheduled`
with `jobs` incrementing = a crash + retry (`max_retries 1`). Crash timings seen:
- ~44 s → `oe-eval-internal` guard (fixed).
- ~5 min (learner ZeRO-3 init) → attention crash: fa4/SP `ValueError` *or* fa3 `no kernel image`.
- Past weight-sync + `accumulate_inference_batches training_step=0` = **success path** (what
  flash_2 achieved). fa4 must reach the same.

**Diagnose a crash:** `beaker job logs <first-job-id>`, strip noise, find the real traceback:
```bash
beaker job logs <JOB> 2>/dev/null | grep -aE "Traceback|ValueError|RuntimeError|no kernel image|Ulysses|flash_attention|CUDA error" \
  | grep -viE "socket.send\(\)|Killed Docker|Closing Docker|Starting Docker|timed out after" | tail -40
```
The fa4/SP failure lives in the **learner** (`PolicyTrainerRayProcess.from_pretrained` →
`UlyssesSPAttentionHF.register_with_transformers`).

## 8. Outcomes / decision tree

- **fa4+SP works (exit 0, hits training steps):** drop the `flash_2` override on B300; prep the
  upstream PR (deepspeed bump and/or SP-aware `detect_attn_implementation`). Update
  `CUDA13_MIGRATION.md` §8.
- **fa4 permitted but crashes at runtime** (interface mismatch in Ulysses' all-to-all with fa4's
  `flash_attn.cute` API): revert to `flash_2`, and the real fix becomes a Ulysses *adapter* for
  fa4 (bigger; report upstream). fa2 remains the working B300+SP backend meanwhile.
- **deepspeed bump breaks unrelated things** (API drift hitting `grpo_fast`/`olmo_core_*`, or
  lock conflicts): assess scope; may need coordinated transformers/olmo-core bumps. If too
  invasive, park the bump and keep fa2.

---

## Appendix: file/artifact index

- `env.cuda13.sh`, `gitconfig.nossh` — worktree isolation (source before anything).
- `scripts/train/build_image_and_launch_dirty.sh` — has `--cuda-version 12|13` (build+push+launch).
- `scripts/general_agent/terminal/rl/holmes_smoke_4gpu_cuda13.sh` — the SP=2 DPPO smoke.
- `scripts/general_agent/terminal/rl/local_rl_2gpu_cuda13.sh` — local (non-Beaker) SP=1 smoke.
- `open_instruct/model_utils.py` — `detect_attn_implementation()` (fa4/fa3/fa2/sdpa logic).
- `open_instruct/utils.py` — `GPU_SPECS` (b300 entry), `check_oe_eval_internal()`.
- `open_instruct/launch_utils.py` — `WEKA_CLUSTERS`/`INTERCONNECT_CLUSTERS` (ai2/holmes).
- Image: `shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13`.
- Smoke-saga exp IDs (2026-07-23): `01KY6KQZ…` (oe-eval crash) → `01KY6MFA…` (fa4/SP crash) →
  `01KY6N9F…` (fa3 no-kernel crash) → `01KY6P3A…` (flash_2, success path). fa4 attempt = the next one.
