---
name: launch-terminal-rl
description: Launch (and set up monitoring for) a Terminal / SWERL agentic-RL training job on Beaker — GRPO with a sandboxed bash/python tool in per-task podman containers. Use when the user wants to start/launch/kick off a terminal-RL (swerl/tmax) training run, create a new such launch script (e.g. a different model size), or relaunch one. NOT for analyzing a running/finished run (use analyze-terminal-rl), NOT for terminal-bench evals (use run-terminal-eval), NOT for task/data generation (use tmax-rl-data).
---

# Launch a Terminal-RL training job

Terminal-RL = GRPO on `open_instruct/grpo_fast.py` with `--tools swerl_vanillux_sandbox`
(a bash/python tool running inside a per-task podman container; reward = automated
test suite pass/fail). Launch scripts live in
`scripts/general_agent/terminal/rl/` (canonical base: `qwen35_4b_base_tmax_10k.sh`).

## 0. ALWAYS verify the image mirror is live — BEFORE launch

These jobs pull sandbox images (`python:3.12-slim` + per-task images) through a
docker.io pull-through cache set by `--env MIRROR_URL=<host:port>`. **`MIRROR_URL` is
baked in at submit time** — a wrong/dead mirror can only be fixed by RELAUNCHING (and you
must never stop a running training job without user approval — see the memory
`feedback_ask_before_cancelling_jobs`).

A dead mirror does NOT crash the job: podman silently falls back to authenticated
docker.io, so it hides as "working" while giving ZERO caching + a failed-connection and
failover on every pull (worse with the podman image janitor re-pulling). Launch scripts
routinely carry a **stale** mirror host copied from an older script.

- **DEAD (do not use):** `jupiter-cs-aus-193.reviz.ai2.in:5000`
- **LIVE (AI2 Jupiter, canonical):** `jupiter-cs-aus-137.reviz.ai2.in:5000` — same mirror
  `run-terminal-eval` uses. But do not assume — **always re-check**, mirror hosts move.

```bash
host=jupiter-cs-aus-137.reviz.ai2.in:5000   # or whatever the script's MIRROR_URL is
curl -sS -m8  -o /dev/null -w "v2 %{http_code}\n" "http://$host/v2/"                       # want 200
curl -sS -m20 -o /dev/null -w "manifest %{http_code}\n" \
  -H "Accept: application/vnd.oci.image.index.v1+json" \
  "http://$host/v2/library/python/manifests/3.12-slim"                                     # want 200 (pull-through)
# optional warmth check: curl -sS "http://$host/v2/_catalog" | head -c 300
```
If not 200, fix `MIRROR_URL` in the script before launching.

## 1. GPU sizing (how grpo_fast splits GPUs)

Total GPUs = `--num_nodes` × `--gpus`. They split into **learners** and **vLLM engines**
on DISJOINT GPUs:
- `--num_learners_per_node` is a LIST, one entry per learner node. `8` → `[8]` = 8 learners
  on 1 node; `8 8` → 16 learners on 2 nodes. `sum(list)` = total learner GPUs.
- `--vllm_num_engines N` × `--vllm_tensor_parallel_size` = engine GPUs.
- Must satisfy: `learner_GPUs + engine_GPUs == total_GPUs`, and
  `vllm_num_engines >= sum(learners)//sequence_parallel_size`, and
  `num_unique_prompts_rollout >= vllm_num_engines`. `sequence_parallel_size>1` requires
  `deepspeed_stage 3`. SP must divide total learner GPUs.

Proven splits on 4 nodes (32 GPUs): **4B** = 8 learners + 24 engines, SP=4;
**9B** = 16 learners (`8 8`) + 16 engines, SP=4 (bigger model needs the extra learner
node for ZeRO-3 optimizer states). Rollouts are the bottleneck (long trajectories + sandbox
latency), so favor engines unless training OOMs.

## 1b. Warm-starting / resuming from a prior RL or SFT checkpoint

Two modes, both with non-obvious traps — full playbook in memory
`reference_warmstart_rl_from_checkpoint`:
- **Warm-start** (fresh optimizer, new run): `--model_name_or_path <ckpt>` + a NEW unique `--exp_name`.
  - ⚠️ Use the **CG-converted `_cg`** checkpoint, NOT the raw one — grpo_fast serves the policy via
    vLLM, which can't load `Qwen3_5ForCausalLM` → crashes at vLLM-engine init. Convert first
    (`convert_qwen35_causallm_to_cg.py`, see `reference_qwen35_causallm_to_cg_conversion`).
  - ⚠️ grpo_fast.py:3212 `snapshot_download` rejects local `/weka` paths (HFValidationError). Rebuild
    is currently broken (causal-conv1d wheel 404), so overlay the patched file at runtime: prepend
    `cp <weka>/patches/grpo_fast_localpath_guard.py open_instruct/grpo_fast.py \&\&` to the mason `--` cmd.
  - ⚠️ **`_build_vlm_name_mapper` (grpo_fast.py:184) only adds vLLM's required `language_model.` prefix
    if the model PATH contains `"qwen3.5"` (with the dot).** A dir named `..._qwen35_..._cg` ("qwen35",
    no dot) → mapper skipped → the trainer→vLLM weight sync crashes `no module named 'model'` and the job
    HANGS after "reached train loop". Fix: broaden the check to match "qwen35"/"qwen36" (in the same
    staged overlay patch), OR name/symlink the checkpoint dir to contain "qwen3.5". This one is the
    biggest time-sink; the same `grpo_fast_localpath_guard.py` overlay carries this fix too.
  - ⚠️ exp_name must not be a PREFIX of another run's (trajectory-glob + wandb contamination).
  - ✅ Validated recipe + failure log in memory `reference_warmstart_rl_from_checkpoint` (run 01KX98RB).
- **Resume state** (continue the SAME run — a faithful continuation, not a warm-start): pass
  `--checkpoint_state_dir <weka .../deletable_checkpoint_states/<user>/<ts>_<rand>>` explicitly
  (mason keeps an explicit /weka path; a fresh mason invocation otherwise auto-assigns a NEW empty
  dir → **silent start-from-scratch**). Restores model + optimizer + LR sched + RNG + **dataset position**.
  - **Dataset position IS preserved** (not reset to example 0): the `ShufflingIterator` restores
    `epoch` (deterministic reshuffle via `seed+epoch` → identical order) + `batches_processed` (skips
    consumed batches) + `excluded_indices`. So it keeps consuming the not-yet-seen prompts in order.
    Requires the SAME parallelism (world size / SP / stage) unless `--deepspeed_checkpoint_load_universal`.
  - **Resuming a run that died / exhausted `max_retries`:** relaunch on the EXISTING image (config-only,
    no rebuild) with that state dir pinned. `keep_last_n_checkpoints` default **3** → relaunch before the
    old `global_step` dirs get pruned.
  - **Verify it took:** leader logs show `Found latest checkpoint: global_stepN`, `Restored episode count`,
    `[DataPreparationActor] Restored state: training_step=N` / `Started preparation loop from training_step=N`,
    and W&B's first logged step ≈ N (not 1). No `Restored…` lines + `training_step=1` = it silently
    started fresh (state dir not pinned). Full detail: **`docs/checkpoint_resumption.md`** (GRPO section).

## 2. Launch

Config-only changes (new script, arg/env edits) do NOT need an image rebuild — reuse the
existing Beaker image (memory `feedback_no_rebuild_for_config_changes`). The dirty launcher
keys the image on the current commit hash and skips the build if it already exists:

```bash
./scripts/train/build_image_and_launch_dirty.sh scripts/general_agent/terminal/rl/<script>.sh
```

It prints `Kicked off Beaker job. https://beaker.org/ex/<EXP_ID>`. Default workspace
`ai2/oe-agents`, priority `urgent`, `--preemptible`, `--max_retries 5`. Docker Hub creds:
`DOCKERHUB_USERNAME=shashankg209` + `--secret DOCKER_PAT=shashankg_DOCKER_PAT`.

Reliability envs worth carrying (from tmax's proven 9B script): `PYTORCH_ALLOC_CONF=expandable_segments:True`,
the podman image-janitor trio (`SWERL_PODMAN_IMAGE_JANITOR_ENABLED/INTERVAL_S=60/UNTIL=10m` —
mitigates the disk-full sandbox hang), `SWERL_RESET_FAILURE_ZERO_REWARD=1`, 8 podman shards.
Note β=0 GRPO can KL-diverge late (~step 200 on the 4B) — watch `objective/kl2_avg`.

## 3. Set up monitoring after launch

Only ONE training job at a time. After launch, arm two `Monitor`s on the EXP_ID:
1. **State** — poll `beaker experiment get <EXP> --format json`, emit on job-state changes
   (scheduled→RUNNING→DONE+exitCode), stop when all finalized. This is the reliable crash detector.
2. **Progress + hang watchdog** — count the per-step `🗡️ Training` banner in the leader log;
   emit progress every ~10 steps; alert if no new step for >30 min while RUNNING (hang).

**Do NOT keyword-grep the logs for errors** (Traceback / Killed / "timed out" / OOM): terminal-RL
rollout logs are full of these because the agent runs arbitrary shell+python in sandboxes
(`Killed Docker container` = normal auto-remove; `Step 'bash' timed out after 120s` = normal
test_timeout). Trust only: job exit code + step-progress stall. Then use `analyze-terminal-rl`
for health/wandb analysis.
